import functools
import logging
import matplotlib as mpl
from matplotlib import _api, _mathtext
from matplotlib.ft2font import LOAD_NO_HINTING
from matplotlib.font_manager import FontProperties
from ._mathtext import (  # noqa: reexported API

        Parse the given math expression *s* at the given *dpi*.  If *prop* is
        provided, it is a `.FontProperties` object specifying the "default"
        font to use in the math expression, used for all non-math text.

        The results are cached, so multiple calls to `parse`
        with the same expression should be fast.

        Depending on the *output* type, this returns either a `VectorParse` or
        a `RasterParse`.
        