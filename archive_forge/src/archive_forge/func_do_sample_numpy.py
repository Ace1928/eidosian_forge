from functools import singledispatch
from sympy.external import import_module
from sympy.stats.crv_types import BetaDistribution, ChiSquaredDistribution, ExponentialDistribution, GammaDistribution, \
from sympy.stats.drv_types import GeometricDistribution, PoissonDistribution, ZetaDistribution
from sympy.stats.frv_types import BinomialDistribution, HypergeometricDistribution
@singledispatch
def do_sample_numpy(dist, size, rand_state):
    return None