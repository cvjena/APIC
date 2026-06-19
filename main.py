"""
Training execution pipeline for APIC (Amortized Physics-Informed Calibration).
Supports dynamic configuration of model architectures and training hyperparameters
via command-line arguments.
"""

import argparse
import os
from typing import Tuple
import numpy as np
import torch

from dataset import AdvDiff1DGenerator, get_cnp_context
from models import get_encoder, CHead, map_to_logtheta
from solver import solve_advdiff_react_spectral

# Basic default evaluation metric fallbacks if external library isn't loaded
def gaussian_nll_loss(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    precision = torch.exp(-logvar)
    return 0.5 * torch.mean(logvar + precision * (target - mean)**2)


@torch.no_grad()
def eval_on_batch(gen, enc, chead, log_lo, log_hi, device, B=512) -> Tuple[float, ...]:
    """Computes rigorous validation metrics across an evaluation batch."""
    enc.eval()
    chead.eval()
    U_obs, theta_true, u0, c_true = gen.batch_true(B=B, device=device)
    t_c, x_c, u_c = get_cnp_context(U_obs, gen)

    z_mean, z_logvar, raw = enc(t_c, x_c, u_c)
    log_theta_hat = map_to_logtheta(raw, log_lo, log_hi)
    theta_hat = torch.exp(log_theta_hat)
    _, c_logvar, c_hat = chead(z_mean, log_theta_hat)

    U_nom = solve_advdiff_react_spectral(
        u0=u0, t_grid=gen.t.to(device), v=theta_hat[:, 0:1], kappa=theta_hat[:, 1:2],
        c=torch.zeros_like(c_hat), kvec=gen.kvec.to(device)
    )
    U_corr = solve_advdiff_react_spectral(
        u0=u0, t_grid=gen.t.to(device), v=theta_hat[:, 0:1], kappa=theta_hat[:, 1:2],
        c=c_hat, kvec=gen.kvec.to(device)
    )

    c_var = torch.exp(0.5 * c_logvar)
    U_obs_flat, U_corr_flat = U_obs.reshape(-1), U_corr.reshape(-1)
    c_var_flat = c_var.unsqueeze(-1).expand_as(U_corr).reshape(-1)

    mae_nom = torch.mean(torch.abs(U_nom - U_obs)).item()
    mae_corr = torch.mean(torch.abs(U_corr - U_obs)).item()
    c_mae = torch.mean(torch.abs(c_hat - c_true)).item()

    return mae_nom, mae_corr, c_mae


def train_pipeline(args):
    """Executes the dual-stage training workflow under the parsed configurations."""
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"Executing APIC pipeline on target device: {device.upper()}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Fix seeds for reproducibility in paper results
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Dataset generation setup
    gen = AdvDiff1DGenerator(Nx=args.nx, Nt=args.nt, t_max=1.0, dtype=torch.float32)

    log_lo = torch.log(torch.tensor([gen.v_rng[0], gen.k_rng[0]], device=device)).unsqueeze(0)
    log_hi = torch.log(torch.tensor([gen.v_rng[1], gen.k_rng[1]], device=device)).unsqueeze(0)

    # Initialize requested configuration from Factory
    enc = get_encoder(model_type=args.model_type, z_dim=args.z_dim, hidden=args.hidden_dim).to(device)
    chead = CHead(z_dim=args.z_dim, hidden=args.hidden_dim, c_max=gen.c_max).to(device)

    # --- Phase 1: Nominal Physics Parameter Alignment ---
    print(f"\n[PHASE 1] Initializing inverse parameter pre-training for {args.steps_p1} steps...")
    opt1 = torch.optim.Adam(enc.parameters(), lr=args.lr_p1)
    
    for ep in range(args.steps_p1):
        enc.train()
        U_nom, theta, _ = gen.batch_nominal(B=args.batch_size, device=device)
        t_c, x_c, u_c = get_cnp_context(U_nom, gen)
        
        _, _, raw = enc(t_c, x_c, u_c)
        log_theta_hat = map_to_logtheta(raw, log_lo, log_hi)
        loss = torch.mean((log_theta_hat - torch.log(torch.clamp(theta, min=1e-8))) ** 2)

        opt1.zero_grad()
        loss.backward()
        opt1.step()
        
        if ep % 500 == 0 or ep == args.steps_p1 - 1:
            print(f"  Step {ep:5d} / {args.steps_p1} | Nominal Parameter MSE: {loss.item():.6f}")

    # --- Phase 2: Structural Calibration & Discrepancy Calibration ---
    print(f"\n[PHASE 2] Jointly optimization discrepancy calibrations for {args.steps_p2} steps...")
    opt2 = torch.optim.Adam(list(enc.parameters()) + list(chead.parameters()), lr=args.lr_p2)
    best_val = float("inf")

    for ep in range(args.steps_p2):
        enc.train()
        chead.train()
        
        U_obs, _, u0, _ = gen.batch_true(B=args.batch_size, device=device)
        t_c, x_c, u_c = get_cnp_context(U_obs, gen)
        
        z_mean, z_logvar, raw = enc(t_c, x_c, u_c)
        z = enc.reparam(z_mean, z_logvar)
        log_theta_hat = map_to_logtheta(raw, log_lo, log_hi)
        theta_hat = torch.exp(log_theta_hat)
        _, c_logvar, c_hat = chead(z, log_theta_hat)

        U_pred = solve_advdiff_react_spectral(
            u0=u0, t_grid=gen.t.to(device), v=theta_hat[:, 0:1], kappa=theta_hat[:, 1:2], c=c_hat, kvec=gen.kvec.to(device)
        )

        nll = gaussian_nll_loss(U_pred, c_logvar, U_obs)
        kl = -0.5 * torch.mean(1 + z_logvar - z_mean**2 - z_logvar.exp())
        
        # Nominal structural anchor retention logic
        U_nom, theta_nom, _ = gen.batch_nominal(B=args.batch_size, device=device)
        t_c_n, x_c_n, u_c_n = get_cnp_context(U_nom, gen)
        _, _, raw_nom = enc(t_c_n, x_c_n, u_c_n)
        anchor = torch.mean((map_to_logtheta(raw_nom, log_lo, log_hi) - torch.log(torch.clamp(theta_nom, min=1e-8))) ** 2)

        loss = nll + 1e-3 * kl + args.lam_anchor * anchor
        
        opt2.zero_grad()
        loss.backward()
        opt2.step()

        if ep % 500 == 0 or ep == args.steps_p2 - 1:
            print(f"  Step {ep:5d} / {args.steps_p2} | Reconstruction NLL Loss: {nll.item():.6f}")

        # Regular verification validation checkpoints
        if ep % args.val_interval == 0 or ep == args.steps_p2 - 1:
            mae_nom, mae_corr, c_mae = eval_on_batch(
                gen, enc, chead, log_lo, log_hi, device, B=512
            )
            print(f"    >> [Validation Step {ep}] Trajectory MAE (Nominal: {mae_nom:.5f} | Corrected: {mae_corr:.5f}) | Discrepancy Error (c MAE): {c_mae:.5f}")
            
            # Save the optimal configuration weights
            if mae_corr < best_val:
                best_val = mae_corr
                ckpt_path = os.path.join(args.out_dir, f"best_{args.model_type.lower()}_model.pt")
                torch.save({
                    "enc": enc.state_dict(),
                    "chead": chead.state_dict(),
                    "log_lo": log_lo.detach().cpu(),
                    "log_hi": log_hi.detach().cpu(),
                    "best_val_mae_corr": best_val,
                    "args": vars(args)
                }, ckpt_path)

    print(f"\nOptimization Finished! Best Checkpoint Documented to path directory: {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APIC Framework: Amortized Physics-Informed Calibration Execution Block")
    
    # Structural & Model architecture configurations
    parser.add_argument("--model_type", type=str, default="ANP", choices=["CNP", "LNP", "ANP"],
                        help="Neural Process backbone variants matching the baseline benchmark comparisons.")
    parser.add_argument("--z_dim", type=int, default=64, help="Dimension size of the target system latent vectors.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Width resolution of inner multilayer perceptron paths.")
    
    # Optimization steps and schedule scales
    parser.add_argument("--steps_p1", type=int, default=2000, help="Iteration updates assigned to baseline parameter alignment.")
    parser.add_argument("--steps_p2", type=int, default=4000, help="Iteration optimization updates allocated to calibration matching.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch constraints allocated per trajectory update loop.")
    parser.add_argument("--lr_p1", type=float, default=1e-3, help="Phase 1 parameter initialization learning rate factor.")
    parser.add_argument("--lr_p2", type=float, default=3e-4, help="Phase 2 discrepancy training learning rate parameter.")
    parser.add_argument("--lam_anchor", type=float, default=2.0, help="Regularization penalty strength balancing baseline retention.")
    
    # Dataset and resolution variables
    parser.add_argument("--nx", type=int, default=64, help="Spatial numerical grid sample boundaries allocation resolution.")
    parser.add_argument("--nt", type=int, default=64, help="Temporal discretization steps along simulated system tracks.")
    
    # System execution environment adjustments
    parser.add_argument("--out_dir", type=str, default="./advdiff_ckpts", help="Directory destination hosting saved tracking states.")
    parser.add_argument("--val_interval", type=int, default=500, help="Steps mapping how frequently to log full tracking evaluations.")
    parser.add_argument("--seed", type=int, default=42, help="Numerical random state seed anchoring pipeline repeatability.")
    parser.add_argument("--no_cuda", action="store_true", help="Bypasses standard hardware GPU scaling if flags are provided manually.")

    parsed_args = parser.parse_args()
    train_pipeline(parsed_args)