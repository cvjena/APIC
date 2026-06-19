from typing import Tuple
import numpy as np
import torch
from solver import solve_advdiff_react_spectral

class AdvDiff1DGenerator:
    """Generates nominal and true trajectories for the 1D Advection-Diffusion PDE."""

    def __init__(self, Nx: int = 64, Nt: int = 64, t_max: float = 1.0, dtype=torch.float32):
        self.Nx = int(Nx)
        self.Nt = int(Nt)
        self.t_max = float(t_max)
        self.dtype = dtype

        self.x = torch.arange(self.Nx, dtype=self.dtype) / float(self.Nx)
        self.t = torch.linspace(0.0, self.t_max, self.Nt, dtype=self.dtype)

        self.v_rng = (0.3, 2.0)
        self.k_rng = (0.002, 0.08)

        self.a = torch.randn(2, dtype=self.dtype) * 0.8
        self.b = torch.tensor(-1.0, dtype=self.dtype)
        self.c_max = 1.0

        kfreq = torch.fft.fftfreq(self.Nx, d=1.0 / self.Nx)
        self.kvec = (2.0 * np.pi) * kfreq.to(self.dtype)

    def sample_theta(self, B: int, device: str) -> torch.Tensor:
        """Samples parameter vectors theta = [v, kappa]."""
        def urand(lo: float, hi: float) -> torch.Tensor:
            return torch.rand(B, 1, device=device, dtype=self.dtype) * (hi - lo) + lo
        return torch.cat([urand(*self.v_rng), urand(*self.k_rng)], dim=1)

    def sample_u0(self, B: int, device: str) -> torch.Tensor:
        """Generates smooth multi-modal random initial fields via mixed Gaussians."""
        x = self.x.to(device).view(1, -1)
        u0 = torch.zeros(B, self.Nx, device=device, dtype=self.dtype)
        n_bumps = torch.randint(low=2, high=4, size=(B,), device=device)

        for i in range(B):
            nb = int(n_bumps[i].item())
            for _ in range(nb):
                amp = torch.rand(1, device=device, dtype=self.dtype) * 1.0 + 0.3
                ctr = torch.rand(1, device=device, dtype=self.dtype)
                sig = torch.rand(1, device=device, dtype=self.dtype) * 0.08 + 0.03
                
                dx = torch.abs(x - ctr)
                dx = torch.minimum(dx, 1.0 - dx)
                u0[i] += amp * torch.exp(-0.5 * (dx / sig) ** 2).squeeze(0)

        u0 = u0 / (u0.amax(dim=1, keepdim=True) + 1e-6)
        return 0.2 + 0.8 * u0

    def true_c(self, theta: torch.Tensor) -> torch.Tensor:
        """Computes structural discrepancy coefficients c."""
        dev = theta.device
        log_theta = torch.log(torch.clamp(theta, min=1e-8))
        a = self.a.to(dev)
        b = self.b.to(dev)
        c = torch.nn.functional.softplus((log_theta * a.view(1, -1)).sum(dim=1, keepdim=True) + b)
        return torch.clamp(c, 0.0, self.c_max)

    def batch_nominal(self, B: int, device: str, noise_std: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates idealized trajectories missing the structural reaction component."""
        theta = self.sample_theta(B, device)
        u0 = self.sample_u0(B, device)
        c = torch.zeros(B, 1, device=device, dtype=self.dtype)

        U = solve_advdiff_react_spectral(
            u0=u0, t_grid=self.t.to(device), v=theta[:, 0:1], kappa=theta[:, 1:2], c=c, kvec=self.kvec.to(device)
        )
        if noise_std > 0:
            U += noise_std * torch.randn_like(U)
        return U, theta, u0

    def batch_true(self, B: int, device: str, noise_std: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates ground truth data governed by the complete hidden system."""
        theta = self.sample_theta(B, device)
        u0 = self.sample_u0(B, device)
        c_true = self.true_c(theta)

        U = solve_advdiff_react_spectral(
            u0=u0, t_grid=self.t.to(device), v=theta[:, 0:1], kappa=theta[:, 1:2], c=c_true, kvec=self.kvec.to(device)
        )
        if noise_std > 0:
            U += noise_std * torch.randn_like(U)
        return U, theta, u0, c_true


def get_cnp_context(
    U_obs: torch.Tensor,
    gen: AdvDiff1DGenerator,
    min_pts: int = 100,
    max_pts: int = 500
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniformly samples sparse space-time context points across a batch."""
    B, Nt, Nx = U_obs.shape
    device = U_obs.device
    num_pts = torch.randint(min_pts, max_pts, (1,)).item()
    
    t_grid = gen.t.to(device)
    x_grid = gen.x.to(device)
    tt, xx = torch.meshgrid(t_grid, x_grid, indexing='ij')

    t_list, x_list, u_list = [], [], []
    for i in range(B):
        t_idx = torch.randint(0, Nt, (num_pts,))
        x_idx = torch.randint(0, Nx, (num_pts,))
        t_list.append(tt[t_idx, x_idx].unsqueeze(-1))
        x_list.append(xx[t_idx, x_idx].unsqueeze(-1))
        u_list.append(U_obs[i, t_idx, x_idx].unsqueeze(-1))
        
    return torch.stack(t_list), torch.stack(x_list), torch.stack(u_list)