import numpy as np
def pdos(self, energy):
    """Projected density of states -1/pi Im(SGS/S)"""
    if self.S is None:
        return -self.retarded(energy).imag.diagonal() / np.pi
    else:
        S = self.S
        SGS = np.dot(S, self.apply_retarded(energy, S))
        return -(SGS.diagonal() / S.diagonal()).imag / np.pi