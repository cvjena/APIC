# APIC: Amortized Physics-Informed Calibration using Neural Processes

This repository contains the official implementation for the paper **"APIC: Amortized Physics-Informed Calibration using Neural Processes"**. 

APIC expands the classical Kennedy-O'Hagan (KOH) framework by incorporating an amortized inference backbone to rapidly calibrate physical parameters and identify state-dependent structural discrepancies across a collection of related physics systems. 

---

## Framework Overview

APIC utilizes a dual-latent architecture to decouple system parameters $\theta$ from unmodeled structural physics $c$. This repository contains the  implementation for the **1D Advection-Diffusion-Reaction PDE** system for three different Neural Process backbones:
* **APIC-CNP:** Conditional Neural Process with deterministic mean aggregation.
* **APIC-LNP:** Latent Neural Process incorporating stochastic variational latent variables.
* **APIC-ANP:** Attentive Neural Process using target-dependent cross-attention.


## Getting Started

1. Installation

```
git clone [https://github.com/your-username/apic-calibration.git](https://github.com/your-username/apic-calibration.git)
cd apic-calibration
pip install -r requirements.txt
```

2. Training a model

You can run experiments using `main.py`. By default, running the script trains the APIC-ANP backbone model:

```
python main.py --model_type ANP --steps_p1 2000 --steps_p2 4000
```

The training loop proceeds in two stages:

Phase 1 (Nominal Pre-training): Learns inverse parameter estimation mappings purely on nominal datasets where $c = 0$.

Phase 2 (Joint Discrepancy Calibration): Calibrates parameters alongside a learned variance head under hidden physical discrepancies.




If you use this code or framework in your research, please consider citing our work:

@article{venkataramanan2026apic,
  title={APIC: Amortized Physics-Informed Calibration using Neural Processes},
  author={Venkataramanan, Aishwarya and Vemuri, Sai Karthikeya and Denzler, Joachim},
  journal={arXiv preprint arXiv:2606.03355},
  year={2026}
}