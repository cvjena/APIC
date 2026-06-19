import torch

def solve_advdiff_react_spectral(
    u0: torch.Tensor,
    t_grid: torch.Tensor,
    v: torch.Tensor,
    kappa: torch.Tensor,
    c: torch.Tensor,
    kvec: torch.Tensor
) -> torch.Tensor:
    """Solves the 1D Advection-Diffusion-Reaction PDE via Fourier Spectral Methods."""
    B, Nx = u0.shape
    Nt = t_grid.shape[0]

    u0_hat = torch.fft.fft(u0.to(torch.complex64), dim=-1)

    k2 = (kvec**2).view(1, Nx)
    k1 = kvec.view(1, Nx)

    v_f32 = v.to(torch.float32)
    k_f32 = kappa.to(torch.float32)
    c_f32 = c.to(torch.float32)

    lam_real = -(k_f32 * k2) - c_f32
    lam_imag = -(v_f32 * k1)
    lam = lam_real.to(torch.complex64) + 1j * lam_imag.to(torch.complex64)

    tt = t_grid.view(1, Nt, 1).to(torch.float32)
    E = torch.exp(tt * lam.view(B, 1, Nx))

    U_hat = u0_hat.view(B, 1, Nx) * E
    U = torch.fft.ifft(U_hat, dim=-1).real
    return U