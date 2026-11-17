# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:09:49 2026

@author: DELL
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from scipy.special import eval_hermite, factorial

# Physical constants
# hbar^2 / 2m for electron in eV nm^2
C = 0.0380998

# Parameters
hbar = 1.0
omega = 1.0

# Choose dimensionless units:
# hbar = 1, m = 1, omega = 1
m = 1.0

# Spatial grid
N = 1000
xmax = 8.0

x = np.linspace(-xmax, xmax, N)
dx = x[1] - x[0]

x_in = x[1:-1]
N_in = len(x_in)

# Harmonic oscillator potential
V = 0.5 * m * omega**2 * x_in**2

# Hamiltonian matrix
diagonal = (
    hbar**2 / (m * dx**2) + V
)

off_diagonal = np.full(
    N_in - 1,
    -hbar**2 / (2*m*dx**2)
)

# Numerical eigenvalues and eigenvectors
num_states = 5

E_num, psi_num = eigh_tridiagonal(
    diagonal,
    off_diagonal,
    select='i',
    select_range=(0, num_states-1)
)

# Normalize numerical wavefunctions
for i in range(num_states):

    psi_num[:, i] /= np.sqrt(
        np.trapezoid(psi_num[:, i]**2, x_in)
    )

# Analytical energy eigenvalues
n = np.arange(num_states)

E_ana = hbar * omega * (n + 0.5)

# Analytical wavefunctions
def analytical_wavefunction(n, x):

    xi = np.sqrt(m*omega/hbar) * x

    normalization = (
        (m*omega/(np.pi*hbar))**0.25
        / np.sqrt(2**n * factorial(n))
    )

    psi = (
        normalization
        * eval_hermite(n, xi)
        * np.exp(-xi**2/2)
    )

    return psi


# Plot wavefunctions
plt.figure(figsize=(10, 7))

for i in range(num_states):

    psi_ana = analytical_wavefunction(n[i], x)

    psi_n = psi_num[:, i]

    # Match the sign of numerical and analytical solutions
    if np.sum(
        psi_n * psi_ana[1:-1]
    ) < 0:
        psi_n = -psi_n

    plt.plot(
        x_in, psi_n,
        label=f"Numerical n={i}"
    )

    plt.plot(
        x, psi_ana,
        '--',
        label=f"Analytical n={i}"
    )

plt.title("Harmonic Oscillator: Wavefunction Comparison")
plt.xlabel("Position x")
plt.ylabel("Wavefunction ψ(x)")
plt.legend()
plt.grid()
plt.show()


# Plot potential and energy levels
plt.figure(figsize=(9, 6))

V_full = 0.5 * m * omega**2 * x**2

plt.plot(
    x, V_full,
    'k',
    linewidth=2,
    label="Potential V(x)"
)

for i in range(num_states):

    plt.axhline(
        E_num[i],
        linestyle='--',
        label=f"Numerical E{i}={E_num[i]:.3f}"
    )

plt.xlabel("Position x")
plt.ylabel("Energy")
plt.title("Harmonic Oscillator Energy Levels")
plt.legend()
plt.grid()
plt.show()


# Energy comparison
print("\nENERGY EIGENVALUE COMPARISON")
print("------------------------------------------")
print("n    Analytical       Numerical       Error")

for i in range(num_states):

    error = abs(
        (E_num[i]-E_ana[i])/E_ana[i]
    )*100

    print(
        f"{i}    {E_ana[i]:.6f}       "
        f"{E_num[i]:.6f}       {error:.6f}%"
    )
