import numpy as np
from scipy.constants import golden as phi
def dicyclic(cls, n, axis=2):
    g1 = cyclic(cls, n, axis).as_rotvec()
    thetas = np.linspace(0, np.pi, n, endpoint=False)
    rv = np.pi * np.vstack([np.zeros(n), np.cos(thetas), np.sin(thetas)]).T
    g2 = np.roll(rv, axis, axis=1)
    return cls.from_rotvec(np.concatenate((g1, g2)))