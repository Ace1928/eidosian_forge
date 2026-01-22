import numpy as np
from collections import namedtuple
from ase.geometry import find_mic
def fit_raw(energies, forces, positions, cell=None, pbc=None):
    """Calculates parameters for fitting images to a band, as for
    a NEB plot."""
    energies = np.array(energies) - energies[0]
    n_images = len(energies)
    fit_energies = np.empty((n_images - 1) * 20 + 1)
    fit_path = np.empty((n_images - 1) * 20 + 1)
    path = [0]
    for i in range(n_images - 1):
        dR = positions[i + 1] - positions[i]
        if cell is not None and pbc is not None:
            dR, _ = find_mic(dR, cell, pbc)
        path.append(path[i] + np.sqrt((dR ** 2).sum()))
    lines = []
    lastslope = None
    for i in range(n_images):
        if i == 0:
            direction = positions[i + 1] - positions[i]
            dpath = 0.5 * path[1]
        elif i == n_images - 1:
            direction = positions[-1] - positions[-2]
            dpath = 0.5 * (path[-1] - path[-2])
        else:
            direction = positions[i + 1] - positions[i - 1]
            dpath = 0.25 * (path[i + 1] - path[i - 1])
        direction /= np.linalg.norm(direction)
        slope = -(forces[i] * direction).sum()
        x = np.linspace(path[i] - dpath, path[i] + dpath, 3)
        y = energies[i] + slope * (x - path[i])
        lines.append((x, y))
        if i > 0:
            s0 = path[i - 1]
            s1 = path[i]
            x = np.linspace(s0, s1, 20, endpoint=False)
            c = np.linalg.solve(np.array([(1, s0, s0 ** 2, s0 ** 3), (1, s1, s1 ** 2, s1 ** 3), (0, 1, 2 * s0, 3 * s0 ** 2), (0, 1, 2 * s1, 3 * s1 ** 2)]), np.array([energies[i - 1], energies[i], lastslope, slope]))
            y = c[0] + x * (c[1] + x * (c[2] + x * c[3]))
            fit_path[(i - 1) * 20:i * 20] = x
            fit_energies[(i - 1) * 20:i * 20] = y
        lastslope = slope
    fit_path[-1] = path[-1]
    fit_energies[-1] = energies[-1]
    return ForceFit(path, energies, fit_path, fit_energies, lines)