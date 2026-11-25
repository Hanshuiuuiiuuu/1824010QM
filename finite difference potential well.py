# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:06:54 2026

@author: DELL
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

# Physical constant: hbar^2 / 2m for electron
# Unit: eV nm^2
C = 0.0380998

# Potential well parameters
a = 0.5          # Half-width in nm
V0 = 10.0        # Potential depth in eV

# Numerical grid
N = 1200
xmax = 5 * a
x = np.linspace(-xmax, xmax, N)
dx = x[1] - x[0]

# Finite square well potential
V = np.where(np.abs(x) <= a, -V0, 0.0)

# ------------------------------------------------
# PART 1: FINITE DIFFERENCE METHOD
# ------------------------------------------------

# Hamiltonian matrix (interior grid points)
V_in = V[1:-1]

diagonal = 2*C/dx**2 + V_in
off_diagonal = np.full(N-3, -C/dx**2)

# Numerical eigenvalues and eigenvectors
energies_num, vectors = eigh_tridiagonal(
    diagonal,
    off_diagonal
)

# Select bound states only
mask = (energies_num > -V0) & (energies_num < 0)

energies_num = energies_num[mask]
vectors = vectors[:, mask]

# ------------------------------------------------
# PART 2: ANALYTICAL SOLUTION
# ------------------------------------------------

# Dimensionless well strength
z0 = a * np.sqrt(V0/C)

def find_roots(z0, parity):

    roots = []
    eps = 1e-9

    if parity == "even":
        intervals = [
            (n*np.pi, (n+0.5)*np.pi)
            for n in range(int(z0/np.pi)+2)
        ]

    else:
        intervals = [
            ((n+0.5)*np.pi, (n+1)*np.pi)
            for n in range(int(z0/np.pi)+2)
        ]

    for left, right in intervals:

        lo = left + eps
        hi = min(right - eps, z0)

        if hi <= lo:
            continue

        if parity == "even":

            def f(z):
                return (
                    z*np.tan(z)
                    - np.sqrt(max(0, z0**2-z**2))
                )

        else:

            def f(z):
                return (
                    -z/np.tan(z)
                    - np.sqrt(max(0, z0**2-z**2))
                )

        if f(lo)*f(hi) < 0:
            root = brentq(f, lo, hi)
            roots.append((root, parity))

    return roots


even_roots = find_roots(z0, "even")
odd_roots = find_roots(z0, "odd")

states = sorted(even_roots + odd_roots)

# Analytical energies
energies_ana = np.array([
    -V0 + C*(z/a)**2 for z, parity in states
])

# ------------------------------------------------
# PART 3: ANALYTICAL WAVEFUNCTIONS
# ------------------------------------------------

def analytical_wavefunction(x, z, parity):

    k = z/a
    alpha = np.sqrt(z0**2-z**2)/a

    psi = np.zeros_like(x)

    inside = np.abs(x) <= a
    outside = np.abs(x) > a

    if parity == "even":

        psi[inside] = np.cos(k*x[inside])

        psi[outside] = (
            np.cos(k*a)
            * np.exp(-alpha*(np.abs(x[outside])-a))
        )

    else:

        psi[inside] = np.sin(k*x[inside])

        psi[outside] = (
            np.sign(x[outside])
            * np.sin(k*a)
            * np.exp(-alpha*(np.abs(x[outside])-a))
        )

    # Normalize
    psi = psi / np.sqrt(np.trapz(psi**2, x))

    return psi


# ------------------------------------------------
# PART 4: NORMALIZE NUMERICAL WAVEFUNCTIONS
# ------------------------------------------------

x_in = x[1:-1]

psi_num = []

for i in range(len(energies_num)):

    psi = vectors[:, i]

    psi = psi / np.sqrt(np.trapz(psi**2, x_in))

    psi_num.append(psi)

# ------------------------------------------------
# PART 5: PLOT POTENTIAL AND ENERGY LEVELS
# ------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(x, V, 'k', linewidth=2,
         label="Potential well")

for i, E in enumerate(energies_num):

    plt.axhline(
        E, linestyle='--',
        label=f"Numerical E{i+1}={E:.3f} eV"
    )

plt.xlabel("Position (nm)")
plt.ylabel("Energy (eV)")
plt.title("Finite Potential Well and Energy Levels")
plt.grid()
plt.legend()
plt.show()

# ------------------------------------------------
# PART 6: COMPARE WAVEFUNCTIONS
# ------------------------------------------------

num_states = min(len(states), len(energies_num))

fig, axes = plt.subplots(
    num_states, 1,
    figsize=(10, 3*num_states),
    squeeze=False
)

for i in range(num_states):

    z, parity = states[i]

    # Analytical wavefunction
    psi_ana = analytical_wavefunction(
        x, z, parity
    )

    # Numerical wavefunction
    psi_n = psi_num[i]

    # Match sign
    if np.sum(psi_n*psi_ana[1:-1]) < 0:
        psi_n = -psi_n

    axes[i, 0].plot(
        x, psi_ana,
        'r--', linewidth=2,
        label="Analytical"
    )

    axes[i, 0].plot(
        x_in, psi_n,
        'b', linewidth=1.5,
        label="Numerical"
    )

    axes[i, 0].set_title(
        f"State {i+1}, Energy = {energies_ana[i]:.4f} eV"
    )

    axes[i, 0].set_xlabel("Position (nm)")
    axes[i, 0].set_ylabel("ψ(x)")
    axes[i, 0].legend()
    axes[i, 0].grid()

plt.tight_layout()
plt.show()

# ------------------------------------------------
# PART 7: ENERGY COMPARISON
# ------------------------------------------------

print("\nENERGY EIGENVALUE COMPARISON")
print("-----------------------------------------------")
print("State   Analytical(eV)   Numerical(eV)   Error(%)")

for i in range(num_states):

    error = abs(
        (energies_num[i]-energies_ana[i])
        / energies_ana[i]
    )*100

    print(
        f"{i+1}       {energies_ana[i]:.6f}       "
        f"{energies_num[i]:.6f}       {error:.5f}"
    )
