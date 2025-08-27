# Vibronic Exciton Dimer Simulator (Developer README)

This package contains a QuTiP-based simulation script for vibronic exciton dimers with symmetric and antisymmetric vibrational modes.

## Features

- **Electronic dimer Hamiltonian**: site energy difference (Δε) and electronic coupling (J).
- **Two harmonic vibrational modes**:
  - Symmetric mode coupled via σ_x.
  - Antisymmetric mode coupled via σ_z.
- **Lindblad dissipators**:
  - Pure electronic dephasing (σ_z).
  - Vibrational damping with Bose–Einstein thermal occupation (up/down Lindblad terms).
- **Observables**:
  - Exciton coherence |ρ_{+,-}(t)| after tracing out vibrations.
  - Vibrational phase synchronization R(t) from ⟨b_s⟩, ⟨b_a⟩ phases.
  - (Extended) Simplified toy-2DES waiting-time trace and FFT spectrum.

## Requirements

- Python 3.9+
- [QuTiP](http://qutip.org/) (`pip install qutip`)
- NumPy, Matplotlib

## Usage

### Single run mode
Simulate dynamics of coherence and synchronization with default parameters:

```bash
python qutip_vibronic_dimer.py --mode single_run --out demo
```

Outputs:
- `demo_coherence.png`: exciton coherence amplitude vs time.
- `demo_R.png`: vibrational synchronization order parameter vs time.

### Sweep mode
Sweep antisymmetric damping rate gamma_a relative to gamma_s:

```bash
python qutip_vibronic_dimer.py --mode sweep_gamma --gamma_s 0.5 --gamma_a_list 1 2 4 8 --out sweep
```

Outputs:
- `sweep_coherence_sweep.png`: coherence traces for different gamma_a values.
- `sweep_R_sweep.png`: synchronization order parameter traces.

### Toy 2DES mode (extended)
Compute waiting-time trace S(T) ~ |ρ_{+,-}(T)| and its FFT spectrum:

```bash
python qutip_vibronic_dimer.py --mode two_des --out twoDES
```

Outputs:
- `twoDES_wait_trace.png`: waiting-time beating signal.
- `twoDES_wait_spectrum.png`: FFT spectrum in cm^-1.

## Key Parameters

- `--deps`: site energy difference ε1 - ε2 (cm^-1).
- `--J`: electronic coupling J (cm^-1).
- `--ws`, `--wa`: symmetric/antisymmetric mode frequencies (cm^-1).
- `--gs`, `--ga`: coupling strengths (cm^-1).
- `--gamma_s`, `--gamma_a`: damping rates (cm^-1).
- `--gamma_phi`: pure dephasing rate (cm^-1).
- `--Nphonon`: harmonic oscillator truncation.
- `--Tmax`: maximum time (fs).
- `--dt`: time step (fs).
- `--T`: bath temperature (K).

## Extending the Code

- For non-Markovian environments, implement HEOM or polaron-transformed Redfield.
- For full 2DES, implement third-order response functions with Liouville pathways.

---
