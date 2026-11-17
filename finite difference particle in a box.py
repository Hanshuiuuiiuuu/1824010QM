# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:05:54 2026

@author: DELL
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Physical constants
# hbar^2 / 2m for electron in eV nm^2
C = 0.0380998

# Box parameters
L = 1.0          # Box length in nm
N = 500          # Number of grid points
num_states = 5   # Number of eigenstates

# Spatial grid
x = np.linspace(0, L, N)
dx = x[1] - x[0]

# Interior grid points (boundary values are zero)
x_in = x[1:-1]
N_in = len(x_in)

# Construct Hamiltonian matrix
diagonal = np.full(N_in, 2*C/dx**2)
off_diagonal = np.full(N_in-1, -C/dx**2)

H = (np.diag(diagonal)
     + np.diag(off_diagonal, 1)
     + np.diag(off_diagonal, -1))

# Numerical eigenvalues and eigenvectors
energies, wavefunctions = eigh(
    H, subset_by_index=[0, num_states-1]
)

# Analytical energy eigenvalues
n = np.arange(1, num_states+1)

E_analytical = C * (n*np.pi/L)**2

# Analytical wavefunctions
psi_analytical = np.sqrt(2/L) * np.sin(
    n[:, None] * np.pi * x / L
)

# Plot numerical and analytical wavefunctions
plt.figure(figsize=(10, 7))

for i in range(num_states):

    # Numerical wavefunction
    psi_num = wavefunctions[:, i]

    # Normalize numerical wavefunction
    psi_num = psi_num / np.sqrt(
        np.trapezoid(psi_num**2, x_in)
    )

    # Match sign with analytical solution
    psi_ana = psi_analytical[i, 1:-1]

    if np.sum(psi_num * psi_ana) < 0:
        psi_num = -psi_num

    # Plot both solutions
    plt.plot(
        x_in, psi_num,
        label=f"Numerical n={i+1}"
    )

    plt.plot(
        x, psi_analytical[i],
        '--',
        label=f"Analytical n={i+1}"
    )

plt.title("Particle in a Box: Numerical vs Analytical")
plt.xlabel("Position x (nm)")
plt.ylabel("Wavefunction ψ(x)")
plt.legend()
plt.grid()
plt.show()


# Plot energy eigenvalues comparison
plt.figure(figsize=(8, 5))

plt.plot(n, E_analytical, 'bo-', label="Analytical")
plt.plot(n, energies, 'r^--', label="Numerical")

plt.xlabel("Quantum Number n")
plt.ylabel("Energy (eV)")
plt.title("Energy Eigenvalue Comparison")
plt.legend()
plt.grid()
plt.show()


# Print energy comparison table
print("\nEnergy Eigenvalue Comparison")
print("----------------------------------------------")
print("n    Analytical(eV)    Numerical(eV)    Error(%)")

for i in range(num_states):

    error = abs(
        (energies[i] - E_analytical[i])
        / E_analytical[i]
    ) * 100

    print(
        f"{n[i]}    {E_analytical[i]:.6f}       "
        f"{energies[i]:.6f}       {error:.5f}"
    )
