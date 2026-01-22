import numpy as np
from numpy import linalg
from ase.transport.selfenergy import LeadSelfEnergy, BoxProbe
from ase.transport.greenfunction import GreenFunction
from ase.transport.tools import subdiagonalize, cutcoupling, dagger,\
from ase.units import kB
def plot_pl_convergence(self):
    self.initialize()
    pl1 = len(self.input_parameters['h1']) // 2
    hlead = self.selfenergies[0].h_ii.real.diagonal()
    hprincipal = self.greenfunction.H.real.diagonal[:pl1]
    import pylab as pl
    pl.plot(hlead, label='lead')
    pl.plot(hprincipal, label='principal layer')
    pl.axis('tight')
    pl.show()