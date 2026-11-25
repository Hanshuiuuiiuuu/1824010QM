
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

e = 3.795
hbar_c = 1973.0
mc2 = 0.511e6

e2 = e**2

# hbar^2 / 2m (eV Angstrom^2)
K = hbar_c**2 / (2 * mc2)

# Bohr radius (Angstrom)
a0 = hbar_c**2 / (mc2 * e2)

print("Bohr radius =", a0, "Angstrom")

r_max = 30.0
N = 6000

r_full = np.linspace(0, r_max, N + 1)
dr = r_full[1] - r_full[0]
r = r_full[1:-1]
n_points = len(r)

V = -e2 / r

main_diag = -2 * np.ones(n_points)
off_diag = np.ones(n_points - 1)

D2 = diags(
    [off_diag, main_diag, off_diag],
    [-1, 0, 1]
) / dr**2

# Hamiltonian for s-wave (l = 0)

H = -K * D2 + diags(V, 0)

energies, eigenvectors = eigsh(
    H, k=2, which='SA', tol=1e-10
)

# Sort energies

idx = np.argsort(energies)

energies = energies[idx]
eigenvectors = eigenvectors[:, idx]

# Normalize wavefunctions

for i in range(2):

    norm = np.sqrt(np.sum(eigenvectors[:, i]**2) * dr)

    eigenvectors[:, i] /= norm

E1 = -13.6
E2 = -13.6 / 4

analytical = np.array([E1, E2])

print("\nENERGY EIGENVALUE COMPARISON")
print("----------------------------------------")
print("State    Numerical (eV)   Analytical (eV)")
print("----------------------------------------")

for i in range(2):

    print(
        f"{i+1}        "
        f"{energies[i]:12.6f}     "
        f"{analytical[i]:12.6f}"
    )

u1 = r * np.exp(-r/a0)

u2 = r * (1 - r/(2*a0)) * np.exp(-r/(2*a0))

u1 /= np.sqrt(np.sum(u1**2) * dr)
u2 /= np.sqrt(np.sum(u2**2) * dr)

analytical_wf = [u1, u2]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i in range(2):

    numerical_wf = eigenvectors[:, i].copy()
    exact_wf = analytical_wf[i].copy()

    # Match arbitrary eigenfunction signs
    if np.sum(numerical_wf * exact_wf) < 0:
        numerical_wf *= -1

    axes[i].plot(
        r, numerical_wf,
        label='Numerical (FDM)',
        linewidth=2
    )

    axes[i].plot(
        r, exact_wf, '--',
        label='Analytical',
        linewidth=2
    )

    axes[i].set_xlabel('r (Angstrom)')
    axes[i].set_ylabel('Reduced radial wavefunction u(r)')
    axes[i].set_title(f"Hydrogen state n = {i+1}, l = 0")
    axes[i].legend()
    axes[i].grid()

plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))

for i in range(2):

    plt.plot(
        r,
        eigenvectors[:, i]**2,
        label=f'n = {i+1}'
    )

plt.xlabel('Radial distance r (Angstrom)')
plt.ylabel('Radial probability density |u(r)|²')
plt.title('Radial Probability Density of Hydrogen Atom')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))

for i in range(2):

    plt.hlines(
        energies[i], 0, 0.4,
        colors='blue', linewidth=3
    )

    plt.hlines(
        analytical[i], 0.6, 1,
        colors='red', linewidth=3
    )

plt.xticks([0.2, 0.8], ['Numerical', 'Analytical'])
plt.ylabel('Energy (eV)')
plt.title('Hydrogen Atom Energy Levels')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
