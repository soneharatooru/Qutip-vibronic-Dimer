# Vibronic Dimer Simulation Tool (User README)

This software simulates **quantum coherence protection** in a simple light-harvesting dimer system.

## What It Does

- Models a **dimer of two pigments** with vibrational modes.
- Shows how **symmetric vs. antisymmetric vibrations** affect coherence.
- Demonstrates how **phase synchronization** of vibrations can extend coherence lifetimes.
- Provides a **toy 2D electronic spectroscopy** signal to visualize quantum beats.

## Quickstart

1. Install requirements:

```bash
pip install qutip numpy matplotlib
```

2. Run a default simulation:

```bash
python qutip_vibronic_dimer.py --mode single_run --out demo
```

Check the generated plots:
- Exciton coherence decay.
- Vibrational synchronization.

3. Try the sweep mode to see how stronger damping of antisymmetric modes protects coherence:

```bash
python qutip_vibronic_dimer.py --mode sweep_gamma --out sweep
```

4. Visualize vibronic beat frequencies with a toy 2DES trace and spectrum:

```bash
python qutip_vibronic_dimer.py --mode two_des --out twoDES
```

## Outputs

- **Coherence plots**: show how long quantum coherence lasts.
- **Synchronization plots**: measure vibrational phase alignment.
- **2DES toy plots**: show beating signals and their frequency spectrum.

---
