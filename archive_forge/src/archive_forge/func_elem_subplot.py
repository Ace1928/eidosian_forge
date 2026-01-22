import os
import numpy as np
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from ase.units import Bohr, Hartree
def elem_subplot(self, curvex, curvey, xlabel, ylabel, name, plt):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i in np.arange(self.Nelements):
        label = name + ' ' + self.elements[i]
        plt.plot(curvex, curvey[i](curvex), label=label)
    plt.legend()