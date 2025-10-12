import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TabDDPM:
    def __init__(self, 
                 dim=[512, 1024, 1024, 512], 
                 num_timesteps=1000,
                 beta_schedule='linear',
                 device='cuda'):
        self.dim = dim
        self.num_timesteps = num_timesteps
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.beta_schedule = beta_schedule
        
        # Initialize noise schedule
        self.betas = self._get_betas().to(self.device)
        self.alphas = (1.0 - self.betas).to(self.device)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(self.device)
        
    def _get_betas(self):
        if self.beta_schedule == 'linear':
            return torch.linspace(1e-4, 0.02, self.num_timesteps)
        elif self.beta_schedule == 'cosine':
            s = 0.008
            steps = self.num_timesteps + 1
            x = torch.linspace(0, self.num_timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            return torch.linspace(1e-4, 0.02, self.num_timesteps)

class MLPDiffusion(nn.Module):
    def __init__(self, input_dim, dim_layers):
        super().__init__()
        
        self.input_dim = input_dim
        self.time_embed_dim = 64
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            nn.ReLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )
        
        # Main network
        layers = []
        prev_dim = input_dim + self.time_embed_dim
        
        for dim in dim_layers:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, input_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x, t):
        # Time embedding
        t_embed = self.time_mlp(t.float().unsqueeze(-1))
        
        # Concatenate input and time embedding
        x_t = torch.cat([x, t_embed], dim=-1)
        
        return self.network(x_t)

class TabDDPMSynthesizer:
    def __init__(self, 
                 dim=[512, 1024, 1024, 512],
                 num_timesteps=1000,
                 lr=1e-3,
                 batch_size=1024,
                 epochs=1000):
        
        self.dim = dim
        self.num_timesteps = num_timesteps
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.ddpm = TabDDPM(dim=dim, num_timesteps=num_timesteps, device='cuda')
        
        print(f"Using device: {self.device}")
        
    def _preprocess_data(self, data):
        """Preprocess tabular data for DDPM training"""
        
        # Separate numerical and categorical columns
        self.numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        processed_data = data.copy()
        self.preprocessors = {}
        self.input_dims = {}
        
        # Process numerical columns with quantile transformation
        if self.numerical_columns:
            self.preprocessors['numerical'] = QuantileTransformer(
                n_quantiles=1000, 
                output_distribution='normal',
                subsample=100000,
                random_state=42
            )
            processed_data[self.numerical_columns] = self.preprocessors['numerical'].fit_transform(
                processed_data[self.numerical_columns]
            )
            self.input_dims['numerical'] = len(self.numerical_columns)
        
        # Process categorical columns with one-hot encoding
        self.categorical_preprocessors = {}
        categorical_data = []
        
        for col in self.categorical_columns:
            # Handle missing values
            processed_data[col] = processed_data[col].fillna('Unknown')
            
            # Label encode first
            le = LabelEncoder()
            encoded = le.fit_transform(processed_data[col])
            
            # FIXED: Convert to numpy array explicitly
            encoded = np.array(encoded, dtype=int)
            n_categories = len(le.classes_)
            
            # Create one-hot encoding with explicit types
            one_hot = np.zeros((encoded.shape[0], n_categories), dtype=np.float32)
            indices = np.arange(encoded.shape[0], dtype=int)
            one_hot[indices, encoded] = 1
            
            self.categorical_preprocessors[col] = {
                'label_encoder': le,
                'n_categories': n_categories
            }
            
            categorical_data.append(one_hot)
        
        # Combine all features
        final_data = processed_data[self.numerical_columns].values.astype(np.float32)
        
        if categorical_data:
            categorical_array = np.concatenate(categorical_data, axis=1)
            final_data = np.concatenate([final_data, categorical_array], axis=1)
        
        self.total_dim = final_data.shape[1]
        return torch.FloatTensor(final_data)
    
    def fit(self, data):
        """Train the TabDDPM model"""
        
        print("📊 Preprocessing data...")
        X = self._preprocess_data(data)
        print(f"Input dimensions: {self.total_dim}")
        
        # Initialize model
        self.model = MLPDiffusion(self.total_dim, self.dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"🧠 Training TabDDPM for {self.epochs} epochs...")
        
        self.model.train()
        for epoch in tqdm(range(self.epochs), desc="Training"):
            epoch_loss = 0
            for batch_idx, (x0,) in enumerate(dataloader):
                x0 = x0.to(self.device)
                
                # Sample random timesteps
                t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=self.device)
                
                # Sample noise
                noise = torch.randn_like(x0)
                
                # Forward diffusion (add noise)
                alpha_bar_t = self.ddpm.alpha_bars[t].unsqueeze(-1)
                x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
                
                # Predict noise
                predicted_noise = self.model(x_t, t)
                
                # Calculate loss
                loss = nn.MSELoss()(predicted_noise, noise)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 100 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        print("✅ Training completed!")
    
    def sample(self, num_samples):
        """Generate synthetic samples"""
        
        print(f"📈 Generating {num_samples} synthetic samples...")
        
        self.model.eval()
        
        # Initialize with pure noise
        with torch.no_grad():
            x = torch.randn(num_samples, self.total_dim, device=self.device)
            
            # Reverse diffusion process
            for t in tqdm(reversed(range(self.num_timesteps)), desc="Sampling"):
                t_tensor = torch.full((num_samples,), t, device=self.device)
                
                # Predict noise
                predicted_noise = self.model(x, t_tensor)
                
                # Compute coefficients
                alpha_t = self.ddpm.alphas[t]
                alpha_bar_t = self.ddpm.alpha_bars[t]
                beta_t = self.ddpm.betas[t]
                
                # Update x
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                x = (1 / torch.sqrt(alpha_t)) * (
                    x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
                ) + torch.sqrt(beta_t) * noise
        
        return self._postprocess_samples(x.cpu().numpy())
    
    def _postprocess_samples(self, samples):
        """Convert generated samples back to original format"""
        
        synthetic_data = pd.DataFrame()
        idx = 0
        
        # Process numerical columns
        if self.numerical_columns:
            num_cols = len(self.numerical_columns)
            numerical_data = samples[:, idx:idx+num_cols]
            
            # Inverse transform
            numerical_data = self.preprocessors['numerical'].inverse_transform(numerical_data)
            
            for i, col in enumerate(self.numerical_columns):
                synthetic_data[col] = numerical_data[:, i]
            
            idx += num_cols
        
        # Process categorical columns
        for col in self.categorical_columns:
            n_categories = self.categorical_preprocessors[col]['n_categories']
            
            # Get one-hot encoded data
            one_hot_data = samples[:, idx:idx+n_categories]
            
            # Convert back to categories
            categories = np.argmax(one_hot_data, axis=1)
            
            # Inverse label encode
            le = self.categorical_preprocessors[col]['label_encoder']
            original_categories = le.inverse_transform(categories)
            
            synthetic_data[col] = original_categories
            idx += n_categories
        
        return synthetic_data

# Main execution
def main():
    print("🚀 Starting TabDDPM Synthetic Data Generation")
    
    # Load data
    df = pd.read_csv("heart_preprocessed.csv")
    print(f"📊 Original dataset shape: {df.shape}")
    
    # Initialize and train TabDDPM
    synthesizer = TabDDPMSynthesizer(
        dim=[512, 1024, 1024, 512],  # Network architecture
        num_timesteps=1000,          # Diffusion steps
        lr=1e-3,                     # Learning rate
        batch_size=512,              # Batch size
        epochs=1000                  # Training epochs
    )
    
    # Fit the model
    synthesizer.fit(df)
    
    # Generate synthetic samples
    synthetic_data = synthesizer.sample(num_samples=100000)
    
    print(f"✅ Generated synthetic data shape: {synthetic_data.shape}")
    
    # Save results
    synthetic_data.to_csv("synthetic_tabddpm.csv", index=False)
    print("💾 Synthetic dataset saved as synthetic_tabddpm.csv")
    
    return synthetic_data

if __name__ == "__main__":
    synthetic_data = main()
