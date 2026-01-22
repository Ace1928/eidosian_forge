import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy import array
def generate_princomp(xo, filen='testsave.py'):
    from mlabwrap import mlab
    np.set_printoptions(precision=14, linewidth=100)
    data = HoldIt('data')
    data.xo = xo
    data.save(filename='testsave.py', comment='generated data, divide by 1000')
    res_princomp = HoldIt('princomp1')
    res_princomp.coef, res_princomp.factors, res_princomp.values = mlab.princomp(x, nout=3)
    res_princomp.save(filename=filen, header=False, comment='mlab.princomp(x, nout=3)')
    res_princomp = HoldIt('princomp2')
    res_princomp.coef, res_princomp.factors, res_princomp.values = mlab.princomp(x[:20,], nout=3)
    np.set_printoptions(precision=14, linewidth=100)
    res_princomp.save(filename=filen, header=False, comment='mlab.princomp(x[:20,], nout=3)')
    res_princomp = HoldIt('princomp3')
    res_princomp.coef, res_princomp.factors, res_princomp.values = mlab.princomp(x[:20,] - x[:20,].mean(0), nout=3)
    np.set_printoptions(precision=14, linewidth=100)
    res_princomp.save(filename=filen, header=False, comment='mlab.princomp(x[:20,]-x[:20,].mean(0), nout=3)')