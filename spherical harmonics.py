# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:59:43 2026

@author: DELL
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y

quantum_numbers = [
    (0, 0),
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (2, 2),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3)
]

#Angular coordinates
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)

theta, phi = np.meshgrid(theta, phi)

fig = plt.figure(figsize=(16, 12))

for i, (l, m) in enumerate(quantum_numbers):

    Y = sph_harm_y(l, m, theta, phi)

    probability = np.abs(Y)**2

    r = probability / probability.max()

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    ax = fig.add_subplot(3, 4, i+1, projection='3d')

    ax.plot_surface(
        x, y, z,
        cmap='plasma',
        edgecolor='none',
        alpha=0.9
    )

    ax.set_title(f"l = {l}, m = {m}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_box_aspect([1, 1, 1])

plt.suptitle(
    "Probability Density of Spherical Harmonics",
    fontsize=18
)

plt.tight_layout()
plt.show()
