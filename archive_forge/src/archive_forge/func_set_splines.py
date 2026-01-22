import os
import numpy as np
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from ase.units import Bohr, Hartree
def set_splines(self):
    self.embedded_energy = np.empty(self.Nelements, object)
    self.electron_density = np.empty(self.Nelements, object)
    self.d_embedded_energy = np.empty(self.Nelements, object)
    self.d_electron_density = np.empty(self.Nelements, object)
    for i in range(self.Nelements):
        self.embedded_energy[i] = spline(self.rho, self.embedded_data[i], k=3)
        self.electron_density[i] = spline(self.r, self.density_data[i], k=3)
        self.d_embedded_energy[i] = self.deriv(self.embedded_energy[i])
        self.d_electron_density[i] = self.deriv(self.electron_density[i])
    self.phi = np.empty([self.Nelements, self.Nelements], object)
    self.d_phi = np.empty([self.Nelements, self.Nelements], object)
    for i in range(self.Nelements):
        for j in range(i, self.Nelements):
            self.phi[i, j] = spline(self.r[1:], self.rphi_data[i, j][1:] / self.r[1:], k=3)
            self.d_phi[i, j] = self.deriv(self.phi[i, j])
            if j != i:
                self.phi[j, i] = self.phi[i, j]
                self.d_phi[j, i] = self.d_phi[i, j]