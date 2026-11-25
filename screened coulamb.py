# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 05:07:37 2026

@author: DELL
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

e = 3.795
hbar_c = 1973.0
mc2 = 0.511e6

e2 = e**2

# Kinetic energy coefficient (eV Angstrom^2)
K = hbar_c**2 / (2 * mc2)

print("hbar^2 / 2m =", K, "eV Angstrom^2")

r_max = 60.0
N = 12000

dr = r_max / N
r = np.arange(1, N) * dr

a_values = [3.0, 5.0, 7.0]

energies = []
wavefunctions = []

for a in a_values:

    # Screened Coulomb potential
    V = -(e2 / r) * np.exp(-r / a)

    # Construct Hamiltonian matrix
    main_diag = 2 * K / dr**2 + V
    off_diag = -K / dr**2 * np.ones(N - 2)

    H = diags(
        [off_diag, main_diag, off_diag],
        [-1, 0, 1],
        format='csr'
    )

    # Calculate ground-state energy
    E, psi = eigsh(
        H,
        k=1,
        which='SA',
        tol=1e-10
    )

    energy = E[0]
    u = psi[:, 0]

    # Normalize wavefunction
    norm = np.sqrt(np.sum(u**2) * dr)
    u = u / norm

    # Choose positive sign at maximum amplitude
    if u[np.argmax(np.abs(u))] < 0:
        u = -u

    energies.append(energy)
    wavefunctions.append(u)

print("\nGROUND STATE ENERGY")
print("--------------------------------")

for i, a in enumerate(a_values):

    print(
        f"a = {a:.1f} Angstrom : "
        f"E = {energies[i]:.6f} eV"
    )

print("\nEnergy to three significant digits:")

for a, E in zip(a_values, energies):
    print(f"a = {a:.1f} Angstrom : {E:.3g} eV")

plt.figure(figsize=(10, 6))

for i, a in enumerate(a_values):

    plt.plot(
        r,
        wavefunctions[i],
        linewidth=2,
        label=f'a = {a} Angstrom'
    )

plt.xlabel('Radial distance r (Angstrom)')
plt.ylabel('Reduced radial wavefunction u(r)')
plt.title('Ground State Wavefunctions in Screened Coulomb Potential')

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

for a in a_values:

    V = -(e2 / r) * np.exp(-r / a)

    plt.plot(
        r,
        V,
        label=f'a = {a} Angstrom'
    )

plt.xlabel('Radial distance r (Angstrom)')
plt.ylabel('Potential energy V(r) (eV)')
plt.title('Screened Coulomb Potentials')

plt.xlim(0, 20)
plt.ylim(-16, 1)

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
