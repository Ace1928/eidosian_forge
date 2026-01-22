from . import _multiarray_umath
from ._multiarray_umath import *  # noqa: F403
from ._multiarray_umath import _UFUNC_API, _add_newdoc_ufunc, _ones_like

Create the numpy.core.umath namespace for backward compatibility. In v1.16
the multiarray and umath c-extension modules were merged into a single
_multiarray_umath extension module. So we replicate the old namespace
by importing from the extension module.

