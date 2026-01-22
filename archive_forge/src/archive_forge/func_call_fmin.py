import numpy as np
import scipy.optimize as opt
from ase.optimize.optimize import Optimizer
def call_fmin(self, xtol, ftol, steps):
    opt.fmin_powell(self.f, self.x0(), xtol=xtol, ftol=ftol, maxiter=steps, disp=0, callback=self.callback, direc=self.direc)