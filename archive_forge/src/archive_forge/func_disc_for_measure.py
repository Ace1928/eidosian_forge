import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.contrib.doe import ModelOptionLib
def disc_for_measure(m, nfe=32, block=True):
    """Pyomo.DAE discretization

    Arguments
    ---------
    m: Pyomo model
    nfe: number of finite elements b
    block: if True, the input model has blocks
    """
    discretizer = pyo.TransformationFactory('dae.collocation')
    if block:
        for s in range(len(m.block)):
            discretizer.apply_to(m.block[s], nfe=nfe, ncp=3, wrt=m.block[s].t)
    else:
        discretizer.apply_to(m, nfe=nfe, ncp=3, wrt=m.t)
    return m