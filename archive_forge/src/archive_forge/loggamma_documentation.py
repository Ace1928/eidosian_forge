from cvxpy.atoms.elementwise.entr import entr
from cvxpy.atoms.elementwise.log import log
from cvxpy.atoms.elementwise.maximum import maximum
Elementwise log of the gamma function.

    Implementation has modest accuracy over the full range, approaching perfect
    accuracy as x goes to infinity. For details on the nature of the approximation,
    refer to `CVXPY GitHub Issue #228 <https://github.com/cvxpy/cvxpy/issues/228#issuecomment-544281906>`_.
    