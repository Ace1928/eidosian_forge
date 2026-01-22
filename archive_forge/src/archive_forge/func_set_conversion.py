import contextvars
from functools import singledispatch
import os
from typing import Any
from typing import Optional
import typing
import warnings
from rpy2.rinterface_lib import _rinterface_capi
import rpy2.rinterface_lib.sexp
import rpy2.rinterface_lib.conversion
import rpy2.rinterface
def set_conversion(this_converter):
    """
    Set conversion rules in the conversion module.
    :param this_converter: The conversion rules
    :type this_converter: :class:`Converter`
    """
    converter_ctx.set(this_converter)