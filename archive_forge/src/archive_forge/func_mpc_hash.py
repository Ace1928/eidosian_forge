import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_hash(z):
    if sys.version_info >= (3, 2):
        re, im = z
        h = mpf_hash(re) + sys.hash_info.imag * mpf_hash(im)
        h = h % 2 ** sys.hash_info.width
        return int(h)
    else:
        try:
            return hash(mpc_to_complex(z, strict=True))
        except OverflowError:
            return hash(z)