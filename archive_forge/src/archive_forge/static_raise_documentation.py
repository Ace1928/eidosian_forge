from numba.core import errors, ir, consts
from numba.core.rewrites import register_rewrite, Rewrite

        Rewrite all matching setitems as static_setitems.
        