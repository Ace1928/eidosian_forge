from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.simulator import Simulator
from pyomo.contrib.sensitivity_toolbox.sens import sensitivity_calculation
def initialize_model(m, n_sim, n_nfe, n_ncp):
    vp_profile = {0: 0.75}
    vt_profile = {0: 0.75}
    m.u_input = Suffix(direction=Suffix.LOCAL)
    m.u_input[m.vp] = vp_profile
    m.u_input[m.vt] = vt_profile
    sim = Simulator(m, package='scipy')
    tsim, profiles = sim.simulate(numpoints=n_sim, varying_inputs=m.u_input)
    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, nfe=n_nfe, ncp=n_ncp, scheme='LAGRANGE-RADAU')
    sim.initialize_model()