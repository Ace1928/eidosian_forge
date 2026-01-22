from operator import gt, lt
from .libmp.backend import xrange
from .functions.functions import SpecialFunctions
from .functions.rszeta import RSCache
from .calculus.quadrature import QuadratureMethods
from .calculus.inverselaplace import LaplaceTransformInversionMethods
from .calculus.calculus import CalculusMethods
from .calculus.optimization import OptimizationMethods
from .calculus.odes import ODEMethods
from .matrices.matrices import MatrixMethods
from .matrices.calculus import MatrixCalculusMethods
from .matrices.linalg import LinearAlgebraMethods
from .matrices.eigen import Eigen
from .identification import IdentificationMethods
from .visualization import VisualizationMethods
from . import libmp
def maxcalls(ctx, f, N):
    """
        Return a wrapped copy of *f* that raises ``NoConvergence`` when *f*
        has been called more than *N* times::

            >>> from mpmath import *
            >>> mp.dps = 15
            >>> f = maxcalls(sin, 10)
            >>> print(sum(f(n) for n in range(10)))
            1.95520948210738
            >>> f(10) # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
              ...
            NoConvergence: maxcalls: function evaluated 10 times

        """
    counter = [0]

    def f_maxcalls_wrapped(*args, **kwargs):
        counter[0] += 1
        if counter[0] > N:
            raise ctx.NoConvergence('maxcalls: function evaluated %i times' % N)
        return f(*args, **kwargs)
    return f_maxcalls_wrapped