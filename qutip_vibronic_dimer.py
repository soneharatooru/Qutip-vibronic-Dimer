#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qutip_vibronic_dimer.py
-----------------------
A minimal, **executable** QuTiP script to simulate a vibronic exciton dimer with
symmetric (Q_s) and antisymmetric (Q_a) vibrational modes. It demonstrates how
selective damping of anti-symmetric modes and near-resonant vibronic coupling
can protect exciton coherence via "phase synchronization" of vibrational modes.

Features
========
1) Lindblad master equation on the composite Hilbert space: electronic dimer x two HOs.
2) Electron-vibration couplings:
   - Antisymmetric mode couples to electronic population difference (σ_z).
   - Symmetric mode couples to electronic coupling (σ_x) [Peierls-like].
3) Observables:
   - Exciton-basis coherence amplitude |ρ_{+,-}(t)| (after tracing out vibrations).
   - Vibrational phases φ_s, φ_a from ⟨b_s⟩, ⟨b_a⟩ and a Kuramoto-like order parameter R(t).
4) Ready-to-run plotting of C_ex(t) and R(t).
5) CLI with two modes: single_run and sweep_gamma.

Install
=======
    pip install qutip numpy matplotlib

Usage
=====
    # A single demo run (defaults tuned for room-T like decoherence and near resonance)
    python qutip_vibronic_dimer.py --mode single_run

    # Change parameters (all in inverse centimeters for energies; time in fs)
    python qutip_vibronic_dimer.py --J 100 --deps 120 --ws 200 --wa 210 --gs 40 --ga 70 \
        --gamma_s 0.5 --gamma_a 4.0 --gamma_phi 25 --Nphonon 6 --Tmax 1500 --dt 2.0

    # Sweep gamma_a / gamma_s to visualize synchronization window
    python qutip_vibronic_dimer.py --mode sweep_gamma --gamma_s 0.5 --gamma_a_list 1 2 4 6 8

Units & Conventions
===================
- Energies are in wavenumbers (cm^-1). We convert to angular frequencies via:
      ω[rad/fs] = 2π * c_cmfs * (energy_cm^-1)
  with c_cmfs ≈ 2.99792458e-5 (speed of light in cm/fs).
- Time is in femtoseconds (fs).
- Temperatures enter only via thermal vibrational occupation (optional).

Notes
=====
- This is a **minimal** model, intended for rapid hypothesis testing. For high fidelity,
  consider HEOM / non-Markovian baths and explicit 2DES response-function simulations.
"""

from __future__ import annotations

import argparse
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from typing import Tuple, List

try:
    from qutip import (basis, qeye, tensor, destroy, sigmax, sigmay, sigmaz,
                       ket2dm, Qobj, mesolve, expect, ptrace)
except Exception as e:
    raise SystemExit("QuTiP is required. Install with `pip install qutip`. Original error: %s" % e)

# ----------------------
# Constants / Conversions
# ----------------------
TWOPI = 2.0 * np.pi
c_cmfs = 2.99792458e-5  # speed of light [cm/fs]

def cm1_to_radfs(x_cm1: float) -> float:
    """Convert wavenumber (cm^-1) to angular frequency (rad/fs)."""
    return TWOPI * c_cmfs * x_cm1

# ----------------------
# Electronic dimer sector
# ----------------------
def electronic_site_ops() -> Tuple[Qobj, Qobj, Qobj, Qobj, Qobj]:
    """Return electronic ops in site basis: I, σx, σy, σz, projectors |1><1|, |2><2|."""
    I = qeye(2)
    sx, sy, sz = sigmax(), sigmay(), sigmaz()
    P1 = ket2dm(basis(2, 0))
    P2 = ket2dm(basis(2, 1))
    return I, sx, sy, sz, P1, P2

def electronic_hamiltonian(deps: float, J: float) -> Qobj:
    """
    H_ex in site basis (cm^-1): H = (deps/2)*σz + J*σx + const.
    deps = ε1 - ε2; J = electronic coupling.
    """
    I, sx, sy, sz, P1, P2 = electronic_site_ops()
    return 0.5 * deps * sz + J * sx

def exciton_eigenvectors(H_ex: Qobj) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return eigenvectors in the site basis as columns of U (2x2 unitary),
    and eigenvalues (ascending). U maps site-basis kets to exciton-basis kets: |ex> = U^\dagger |site>.
    """
    evals, evecs = H_ex.eigenstates()
    # eigenstates() returns sorted evals ascending; evecs are Qobj kets
    U = np.column_stack([v.full().ravel() for v in evecs])  # site->exciton columns
    lams = np.array(evals, dtype=float)
    return U, lams

def project_to_exciton(rho_e_site: Qobj, U: np.ndarray) -> np.ndarray:
    """
    Transform a 2x2 electronic density matrix (site basis) into exciton basis using U from exciton_eigenvectors().
    site -> exciton:  ρ_ex = U^† ρ_site U
    """
    rho = rho_e_site.full()
    return U.conj().T @ rho @ U

# ----------------------
# Vib modes and operators
# ----------------------
def vib_ops(N: int) -> Tuple[Qobj, Qobj, Qobj]:
    """Return a, adag, I for a harmonic oscillator truncated to N levels."""
    a = destroy(N)
    I = qeye(N)
    return a, a.dag(), I

# ----------------------
# Build composite Hamiltonian
# ----------------------
def build_hamiltonian_and_ops(
    deps_cm1: float,
    J_cm1: float,
    ws_cm1: float,
    wa_cm1: float,
    gs_cm1: float,
    ga_cm1: float,
    Nphonon: int
) -> Tuple[Qobj, dict]:
    """
    Construct the full Hamiltonian H = H_ex + H_vib + H_coupling in the composite space:
    H_ex (2) ⊗ HO_s (N) ⊗ HO_a (N)

    - Antisymmetric mode couples via σ_z ⊗ (a_a + a_a†)   (Holstein-like)
    - Symmetric mode couples via σ_x ⊗ (a_s + a_s†)        (Peierls-like)
    """
    I_e, sx, sy, sz, P1, P2 = electronic_site_ops()
    a_s, adag_s, I_s = vib_ops(Nphonon)
    a_a, adag_a, I_a = vib_ops(Nphonon)

    # Frequencies in rad/fs
    ws = cm1_to_radfs(ws_cm1)
    wa = cm1_to_radfs(wa_cm1)

    # Convert couplings to rad/fs
    gs = cm1_to_radfs(gs_cm1)
    ga = cm1_to_radfs(ga_cm1)

    # Electronic Hamiltonian (convert to rad/fs)
    H_ex = cm1_to_radfs(1.0) * electronic_hamiltonian(deps_cm1, J_cm1)  # scale cm^-1 -> rad/fs

    # Vibrational Hamiltonian (use number operators; drop zero-point for simplicity)
    H_vib = (tensor(qeye(2), adag_s * a_s, I_a) * ws) + (tensor(qeye(2), I_s, adag_a * a_a) * wa)

    # Couplings
    Qs = (a_s + adag_s)  # symmetric mode coordinate
    Qa = (a_a + adag_a)  # antisymmetric mode coordinate

    H_c_s = tensor(sx, Qs, I_a) * gs   # symmetric couples via σ_x
    H_c_a = tensor(sz, I_s, Qa) * ga   # antisymmetric couples via σ_z

    H = tensor(H_ex, I_s, I_a) + H_vib + H_c_s + H_c_a

    ops = {
        "I_e": I_e,
        "sx": sx, "sz": sz,
        "a_s": a_s, "a_a": a_a,
        "Qs": Qs, "Qa": Qa,
        "I_s": I_s, "I_a": I_a,
        "H_ex_cm1": electronic_hamiltonian(deps_cm1, J_cm1),  # for reporting in cm^-1
    }
    return H, ops

# ----------------------
# Collapse operators
# ----------------------
def collapse_ops(
    gamma_phi_cm1: float,
    gamma_s_cm1: float,
    gamma_a_cm1: float,
    Nphonon: int,
    T_kelvin: float = 300.0,
    ws_cm1: float = 200.0,
    wa_cm1: float = 210.0
) -> List[Qobj]:
    """
    Build Lindblad collapse operators:
    - Electronic pure dephasing via σ_z.
    - Vibrational damping for each mode with thermal occupation n_th.
    """
    # Convert all rates from cm^-1 to rad/fs
    gphi = cm1_to_radfs(gamma_phi_cm1)
    gs = cm1_to_radfs(gamma_s_cm1)
    ga = cm1_to_radfs(gamma_a_cm1)

    # Vibrational operators
    a_s, adag_s, I_s = vib_ops(Nphonon)
    a_a, adag_a, I_a = vib_ops(Nphonon)

    # Thermal occupations (Bose-Einstein). Using energy in cm^-1: E = hc * (cm^-1).
    # k_B in cm^-1/K ≈ 0.695 cm^-1/K (since 1 cm^-1 ≈ 1.4388 K). More precisely: k_B/hc ≈ 0.69503476 cm^-1/K.
    kB_cm1K = 0.69503476
    n_s = 1.0/(np.exp(ws_cm1/(kB_cm1K*T_kelvin)) - 1.0) if T_kelvin > 0 else 0.0
    n_a = 1.0/(np.exp(wa_cm1/(kB_cm1K*T_kelvin)) - 1.0) if T_kelvin > 0 else 0.0

    # Electronic dephasing (σ_z)
    I_e, sx, sy, sz, P1, P2 = electronic_site_ops()
    c_deph = np.sqrt(gphi) * tensor(sz, qeye(Nphonon), qeye(Nphonon))

    # Vibrational damping (L[b], L[b^†]) for each mode
    c_s_down = np.sqrt(gs*(n_s+1.0)) * tensor(qeye(2), a_s, qeye(Nphonon))
    c_s_up   = np.sqrt(gs*n_s)       * tensor(qeye(2), a_s.dag(), qeye(Nphonon))

    c_a_down = np.sqrt(ga*(n_a+1.0)) * tensor(qeye(2), qeye(Nphonon), a_a)
    c_a_up   = np.sqrt(ga*n_a)       * tensor(qeye(2), qeye(Nphonon), a_a.dag())

    return [c_deph, c_s_down, c_s_up, c_a_down, c_a_up]

# ----------------------
# Initial state
# ----------------------
def initial_state_superposition(H_ex_cm1: Qobj, Nphonon: int) -> Qobj:
    """
    Electronic part: equal superposition of excitons |ψ_e> = (|+> + |->)/√2 (created by diagonalizing H_ex).
    Vibrational: both modes in vacuum |0_s, 0_a>.
    """
    U, _ = exciton_eigenvectors(H_ex_cm1)  # in cm^-1 but only eigenvectors matter
    # Build exciton kets in site basis
    # Columns of U are eigenvectors in site basis for exciton |+>, |-> (ascending energy).
    ket_plus_site = Qobj(U[:, 0].reshape((2, 1)))
    ket_minus_site = Qobj(U[:, 1].reshape((2, 1)))
    # Superposition in site basis
    psi_e = (ket_plus_site + ket_minus_site).unit()
    # Vib ground states
    ket0_s = basis(Nphonon, 0)
    ket0_a = basis(Nphonon, 0)
    return tensor(psi_e, ket0_s, ket0_a)  # pure state

# ----------------------
# Observables & helpers
# ----------------------
def reduced_electronic_dm(rho_full: Qobj) -> Qobj:
    return ptrace(rho_full, 0)  # trace out both HOs

def exciton_coherence_abs(rho_e_site: Qobj, H_ex_cm1: Qobj) -> float:
    U, _ = exciton_eigenvectors(H_ex_cm1)
    rho_ex = project_to_exciton(rho_e_site, U)
    # off-diagonal magnitude |ρ_{12}| in exciton basis
    return np.abs(rho_ex[0, 1])

def vibrational_phases(rho_full: Qobj, Nphonon: int) -> Tuple[float, float]:
    a_s, _, _ = vib_ops(Nphonon)
    a_a, _, _ = vib_ops(Nphonon)
    bs = expect(tensor(qeye(2), a_s, qeye(Nphonon)), rho_full)
    ba = expect(tensor(qeye(2), qeye(Nphonon), a_a), rho_full)
    phi_s = np.angle(bs) if np.abs(bs) > 1e-12 else 0.0
    phi_a = np.angle(ba) if np.abs(ba) > 1e-12 else 0.0
    return float(phi_s), float(phi_a)

def kuramoto_R(phi_s: float, phi_a: float) -> float:
    return float(0.5 * np.abs(np.exp(1j*phi_s) + np.exp(1j*phi_a)))

# ----------------------
# Time evolution
# ----------------------
def simulate(
    deps: float = 120.0,
    J: float = 100.0,
    ws: float = 200.0,
    wa: float = 210.0,
    gs: float = 40.0,
    ga: float = 70.0,
    gamma_s: float = 0.5,
    gamma_a: float = 4.0,
    gamma_phi: float = 25.0,
    Nphonon: int = 6,
    Tmax: float = 1500.0,
    dt: float = 2.0,
    T_kelvin: float = 300.0,
    store_states: bool = False
):
    """
    Run a single trajectory of the Lindblad master equation and return time arrays and observables.
    All energy-like parameters are in cm^-1; rates likewise. Time in fs.

    Returns: tlist, Coherence(t), R(t)
    """
    H, ops = build_hamiltonian_and_ops(deps, J, ws, wa, gs, ga, Nphonon)
    H_ex_cm1 = ops["H_ex_cm1"]

    c_ops = collapse_ops(gamma_phi, gamma_s, gamma_a, Nphonon, T_kelvin, ws, wa)
    psi0 = initial_state_superposition(H_ex_cm1, Nphonon)
    rho0 = ket2dm(psi0)

    tlist = np.arange(0.0, Tmax + dt, dt)

    e_ops = None  # we'll post-process
    result = mesolve(H, rho0, tlist, c_ops, e_ops, progress_bar=None, options=None, args={})

    coh = []
    Rvals = []
    for state in result.states:
        rho_e = reduced_electronic_dm(state)
        coh.append(exciton_coherence_abs(rho_e, H_ex_cm1))

        phi_s, phi_a = vibrational_phases(state, Nphonon)
        Rvals.append(kuramoto_R(phi_s, phi_a))

    return tlist, np.array(coh), np.array(Rvals)

# ----------------------
# Plotting helpers
# ----------------------
def plot_single(tlist: npt.NDArray[np.floating], coh: npt.NDArray[np.floating],
                Rvals: npt.NDArray[np.floating], out_prefix: str = "single_run"):
    plt.figure()
    plt.plot(tlist, coh, label="|rho_{+,-}|")
    plt.xlabel("Time (fs)"); plt.ylabel("Exciton coherence amplitude")
    plt.title("Exciton coherence |rho_{+,-}| vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_coherence.png", dpi=150)

    plt.figure()
    plt.plot(tlist, Rvals, label="R(t)")
    plt.xlabel("Time (fs)"); plt.ylabel("Kuramoto order parameter R(t)")
    plt.title("Vibrational phase synchronization R(t)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_R.png", dpi=150)

def plot_sweep(gamma_as: List[float], tlist: npt.NDArray[np.floating],
               curves: List[npt.NDArray[np.floating]], ylabel: str, out_name: str):
    plt.figure()
    for ga_cm1, y in zip(gamma_as, curves):
        plt.plot(tlist, y, label=f"gamma_a={ga_cm1} cm^-1")
    plt.xlabel("Time (fs)"); plt.ylabel(ylabel)
    plt.title(out_name.replace("_", " "))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_name}.png", dpi=150)

# ----------------------
# CLI / Main
# ----------------------
def main():
    p = argparse.ArgumentParser(description="Vibronic exciton dimer with symmetric/antisymmetric modes (QuTiP).")
    p.add_argument("--mode", choices=["single_run", "sweep_gamma"], default="single_run")
    p.add_argument("--deps", type=float, default=120.0, help="Site-energy difference ε1-ε2 (cm^-1).")
    p.add_argument("--J", type=float, default=100.0, help="Electronic coupling J (cm^-1).")
    p.add_argument("--ws", type=float, default=200.0, help="Symmetric mode frequency ω_s (cm^-1).")
    p.add_argument("--wa", type=float, default=210.0, help="Antisymmetric mode frequency ω_a (cm^-1).")
    p.add_argument("--gs", type=float, default=40.0, help="Symmetric vibronic coupling g_s (cm^-1).")
    p.add_argument("--ga", type=float, default=70.0, help="Antisymmetric vibronic coupling g_a (cm^-1).")
    p.add_argument("--gamma_s", type=float, default=0.5, help="Damping rate of symmetric mode (cm^-1).")
    p.add_argument("--gamma_a", type=float, default=4.0, help="Damping rate of antisymmetric mode (cm^-1).")
    p.add_argument("--gamma_a_list", nargs="*", type=float, default=None,
                   help="List of gamma_a values for sweep (cm^-1).")
    p.add_argument("--gamma_phi", type=float, default=25.0, help="Electronic pure dephasing rate (cm^-1).")
    p.add_argument("--Nphonon", type=int, default=6, help="HO truncation for each mode.")
    p.add_argument("--Tmax", type=float, default=1500.0, help="Max time (fs).")
    p.add_argument("--dt", type=float, default=2.0, help="Time step (fs).")
    p.add_argument("--T", type=float, default=300.0, help="Temperature for vibrational baths (K).")
    p.add_argument("--out", type=str, default="run", help="Output prefix.")
    args = p.parse_args()

    if args.mode == "single_run":
        tlist, coh, Rvals = simulate(
            deps=args.deps, J=args.J, ws=args.ws, wa=args.wa,
            gs=args.gs, ga=args.ga,
            gamma_s=args.gamma_s, gamma_a=args.gamma_a, gamma_phi=args.gamma_phi,
            Nphonon=args.Nphonon, Tmax=args.Tmax, dt=args.dt, T_kelvin=args.T
        )
        plot_single(tlist, coh, Rvals, out_prefix=args.out)
        print("Done. Saved:", f"{args.out}_coherence.png", f"{args.out}_R.png")

    elif args.mode == "sweep_gamma":
        if not args.gamma_a_list:
            args.gamma_a_list = [args.gamma_a, 2*args.gamma_a, 4*args.gamma_a, 8*args.gamma_a]
        coh_curves = []
        R_curves = []
        for ga in args.gamma_a_list:
            tlist, coh, Rvals = simulate(
                deps=args.deps, J=args.J, ws=args.ws, wa=args.wa,
                gs=args.gs, ga=ga,
                gamma_s=args.gamma_s, gamma_a=ga, gamma_phi=args.gamma_phi,
                Nphonon=args.Nphonon, Tmax=args.Tmax, dt=args.dt, T_kelvin=args.T
            )
            coh_curves.append(coh)
            R_curves.append(Rvals)
        plot_sweep(args.gamma_a_list, tlist, coh_curves, "|rho_{+,-}|", f"{args.out}_coherence_sweep")
        plot_sweep(args.gamma_a_list, tlist, R_curves, "R(t)", f"{args.out}_R_sweep")
        print("Done. Saved:", f"{args.out}_coherence_sweep.png", f"{args.out}_R_sweep.png")

if __name__ == "__main__":
    main()
