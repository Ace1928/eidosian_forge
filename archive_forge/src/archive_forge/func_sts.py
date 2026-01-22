import numpy as np
from ase.io.jsonio import read_json, write_json
def sts(self, x, y, z, bias0, bias1, biasstep):
    """Returns the dI/dV curve for position x, y at height z (in Angstrom),
        for bias from bias0 to bias1 with step biasstep."""
    biases = np.arange(bias0, bias1 + biasstep, biasstep)
    I = np.zeros(biases.shape)
    for b in np.arange(len(biases)):
        print(b, biases[b])
        I[b] = self.pointcurrent(biases[b], x, y, z)
    dIdV = np.gradient(I, biasstep)
    return (biases, I, dIdV)