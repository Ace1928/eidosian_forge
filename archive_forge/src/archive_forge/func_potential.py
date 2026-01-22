import numpy as np
def potential(self, x):
    self.atoms.set_positions(x.reshape(-1, 3))
    V = self.atoms.get_potential_energy(force_consistent=True)
    gradV = -self.atoms.get_forces().reshape(-1)
    return np.append(np.array(V).reshape(-1), gradV)