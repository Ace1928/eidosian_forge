import warnings
from ._sf_error import SpecialFunctionWarning, SpecialFunctionError
from . import _ufuncs
from ._ufuncs import *
from ._support_alternative_backends import (
from . import _basic
from ._basic import *
from ._logsumexp import logsumexp, softmax, log_softmax
from . import _orthogonal
from ._orthogonal import *
from ._spfun_stats import multigammaln
from ._ellip_harm import (
from ._lambertw import lambertw
from ._spherical_bessel import (
from . import add_newdocs, basic, orthogonal, specfun, sf_error, spfun_stats
from scipy._lib._testutils import PytestTester
def btdtri(*args, **kwargs):
    warnings.warn(_depr_msg.format('betaincinv'), category=DeprecationWarning, stacklevel=2)
    return _ufuncs.btdtri(*args, **kwargs)