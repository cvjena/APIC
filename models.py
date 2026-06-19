"""
Attentive, Latent, and Conditional Neural Process architectures for APIC.
Disentangles instance-specific physical parameters from structural discrepancies.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def map_to_logtheta(raw: torch.Tensor, log_lo: torch.Tensor, log_hi: torch.Tensor) -> torch.Tensor:
    """Scales raw network outputs to bounded parameter constraints in log-space."""
    return log_lo + torch.sigmoid(raw) * (log_hi - log_lo)


class MultiHeadAttention(nn.Module):
    """Encapsulates PyTorch Multihead Attention to enforce batch_first ordering."""
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.mha(q, k, v)
        return attn_output


class BaseAPICEncoder(nn.Module):
    """Base framework for APIC Encoders handling structural parameter mapping."""
    def __init__(self, z_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.z_dim = z_dim
        self.input_projection = nn.Linear(3, hidden)
        
        # Shared projection mapping back to parameters
        self.to_z = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * z_dim)
        )
        self.theta_head = nn.Linear(z_dim, 2)

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


class CNPEncoder(BaseAPICEncoder):
    """Conditional Neural Process variant. Uses deterministic Mean Aggregation."""
    def forward(self, t_c: torch.Tensor, x_c: torch.Tensor, u_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx = torch.cat([t_c, x_c, u_c], dim=-1) # [B, N, 3]
        h = self.input_projection(ctx)           # [B, N, hidden]
        
        # Permutation-invariant aggregation via Mean Pooling 
        r = torch.mean(h, dim=1)                 # [B, hidden]
        
        z_out = self.to_z(r)
        z_mean, z_logvar = torch.split(z_out, self.z_dim, dim=-1)
        
        
        raw_theta = self.theta_head(z_mean)
        return z_mean, z_logvar, raw_theta


class LNPEncoder(BaseAPICEncoder):
    """Latent Neural Process variant. Uses Mean Aggregation with Variational Latents."""
    def forward(self, t_c: torch.Tensor, x_c: torch.Tensor, u_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx = torch.cat([t_c, x_c, u_c], dim=-1)
        h = self.input_projection(ctx)
        
        # Mean Pooling aggregation
        r = torch.mean(h, dim=1)
        
        z_out = self.to_z(r)
        z_mean, z_logvar = torch.split(z_out, self.z_dim, dim=-1)
        
        # LNP uses stochastic sampling via reparameterization trick
        z = z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_logvar)
        raw_theta = self.theta_head(z)
        return z_mean, z_logvar, raw_theta


class ANPEncoder(BaseAPICEncoder):
    """Attentive Neural Process variant. Uses Attention-driven Context Aggregation."""
    def __init__(self, z_dim: int = 64, hidden: int = 256, num_heads: int = 4):
        super().__init__(z_dim=z_dim, hidden=hidden)
        self.self_attn = MultiHeadAttention(hidden, num_heads)
        self.latent_seed = nn.Parameter(torch.randn(1, 1, hidden))
        self.cross_attn = MultiHeadAttention(hidden, num_heads)

    def forward(self, t_c: torch.Tensor, x_c: torch.Tensor, u_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = t_c.shape[0]
        ctx = torch.cat([t_c, x_c, u_c], dim=-1)
        
        h = self.input_projection(ctx)
        h = h + self.self_attn(h, h, h) 
        
        queries = self.latent_seed.expand(B, -1, -1)
        r = self.cross_attn(queries, h, h).squeeze(1)
        
        z_out = self.to_z(r)
        z_mean, z_logvar = torch.split(z_out, self.z_dim, dim=-1)
        
        z = z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_logvar)
        raw_theta = self.theta_head(z)
        return z_mean, z_logvar, raw_theta


class CHead(nn.Module):
    """Predictive discrepancy head calibration network."""
    def __init__(self, z_dim: int = 64, hidden: int = 256, c_max: float = 1.0):
        super().__init__()
        self.c_max = float(c_max)
        self.net = nn.Sequential(
            nn.Linear(z_dim + 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )
        self.z_dim = z_dim

    def forward(self, z: torch.Tensor, log_theta_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_out = self.net(torch.cat([z, log_theta_hat], dim=1))
        raw_mean, raw_logvar = torch.split(raw_out, 1, dim=-1)
        raw = raw_mean + torch.randn_like(raw_mean) * torch.exp(0.5 * raw_logvar)
        c = torch.clamp(F.softplus(raw), 0.0, self.c_max)
        return raw_mean, raw_logvar, c


def get_encoder(model_type: str, z_dim: int = 64, hidden: int = 256, **kwargs) -> BaseAPICEncoder:
    """Factory function giving downstream scripts instant configuration mapping choices."""
    model_type = model_type.upper()
    if model_type == "CNP":
        return CNPEncoder(z_dim=z_dim, hidden=hidden)
    elif model_type == "LNP":
        return LNPEncoder(z_dim=z_dim, hidden=hidden)
    elif model_type == "ANP":
        return ANPEncoder(z_dim=z_dim, hidden=hidden, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {model_type}. Select from ['CNP', 'LNP', 'ANP']")