import spherogram
import snappy
from sage.all import Link as SageLink
from sage.all import LaurentPolynomialRing, PolynomialRing, ZZ, var
from sage.symbolic.ring import SymbolicRing

    To match our conventions (which seem to agree with KnotAtlas), we
    need to swap q and 1/q.
    