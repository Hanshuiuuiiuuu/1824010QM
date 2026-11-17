# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:02:05 2026

@author: DELL
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

C = 0.0380998

well_parameters = [
    (1.0, 5.0),
    (1.0, 10.0),
    (1.5, 10.0),
    (2.0, 10.0)
]

def find_roots(z0, parity):

    roots = []

    if parity == "even":
        n_values = range(int(z0 / np.pi) + 2)

        for n in n_values:

            left = n * np.pi
            right = (n + 0.5) * np.pi

            lo = left + 1e-8
            hi = min(right - 1e-8, z0)

            if hi <= lo:
                continue

            def equation(z):
                return z * np.tan(z) - np.sqrt(
                    max(0, z0**2 - z**2)
                )

            if equation(lo) * equation(hi) < 0:
                root = brentq(equation, lo, hi)
                roots.append(root)

    else:
        n_values = range(int(z0 / np.pi) + 2)

        for n in n_values:

            left = (n + 0.5) * np.pi
            right = (n + 1) * np.pi

            lo = left + 1e-8
            hi = min(right - 1e-8, z0)

            if hi <= lo:
                continue

            def equation(z):
                return -z / np.tan(z) - np.sqrt(
                    max(0, z0**2 - z**2)
                )

            if equation(lo) * equation(hi) < 0:
                root = brentq(equation, lo, hi)
                roots.append(root)

    return roots


# Wavefunction calculation
def wavefunction(x, a, z, parity, V0):

    k = z / a
    alpha = np.sqrt(z0_global**2 - z**2) / a

    psi = np.zeros_like(x)

    inside = np.abs(x) <= a
    outside = np.abs(x) > a

    if parity == "even":

        psi[inside] = np.cos(k * x[inside])

        psi[outside] = (
            np.cos(k * a)
            * np.exp(-alpha * (np.abs(x[outside]) - a))
        )

    else:

        psi[inside] = np.sin(k * x[inside])

        psi[outside] = (
            np.sign(x[outside])
            * np.sin(k * a)
            * np.exp(-alpha * (np.abs(x[outside]) - a))
        )

    return psi


# Main plotting
fig, axes = plt.subplots(
    len(well_parameters), 2,
    figsize=(14, 16)
)

for i, (a, V0) in enumerate(well_parameters):

    # Dimensionless well strength
    z0_global = a * np.sqrt(V0 / C)

    # Find even and odd roots
    even_roots = find_roots(z0_global, "even")
    odd_roots = find_roots(z0_global, "odd")

    all_states = (
        [(z, "even") for z in even_roots]
        + [(z, "odd") for z in odd_roots]
    )

    # Sort states by energy
    all_states.sort(key=lambda item: item[0])

    print("\nWell width =", 2*a, "nm")
    print("Potential depth =", V0, "eV")

    # Position range
    x = np.linspace(-3*a, 3*a, 1200)

    ax1 = axes[i, 0]
    ax2 = axes[i, 1]

    # Plot potential
    potential = np.where(np.abs(x) <= a, -V0, 0)

    ax1.plot(x, potential, 'k', linewidth=2)
    ax1.set_title(
        f"Potential Well: width={2*a} nm, V0={V0} eV"
    )
    ax1.set_xlabel("Position (nm)")
    ax1.set_ylabel("Potential Energy (eV)")
    ax1.grid()

    # Plot energy levels and wavefunctions
    for j, (z, parity) in enumerate(all_states):

        # Energy eigenvalue
        E = -V0 + C * (z/a)**2

        print(
            f"State {j+1}: {parity}, "
            f"z = {z:.5f}, E = {E:.6f} eV"
        )

        # Calculate wavefunction
        psi = wavefunction(x, a, z, parity, V0)

        # Normalize wavefunction
        norm = np.sqrt(np.trapz(psi**2, x))
        psi = psi / norm

        # Plot normalized wavefunction
        ax2.plot(
            x, psi,
            label=f"n={j+1}, E={E:.3f} eV"
        )

        # Energy level on potential plot
        ax1.axhline(
            E, color='red',
            linestyle='--', alpha=0.7
        )

    ax2.set_title("Normalized Eigenfunctions")
    ax2.set_xlabel("Position (nm)")
    ax2.set_ylabel("Normalized ψ(x)")
    ax2.legend(fontsize=8)
    ax2.grid()

plt.tight_layout()
plt.show()
