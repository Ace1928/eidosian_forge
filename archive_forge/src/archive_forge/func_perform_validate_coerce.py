import base64
import numbers
import textwrap
import uuid
from importlib import import_module
import copy
import io
import re
import sys
import warnings
from _plotly_utils.optional_imports import get_module
@staticmethod
def perform_validate_coerce(v, allow_number=None):
    """
        Validate, coerce, and return a single color value. If input cannot be
        coerced to a valid color then return None.

        Parameters
        ----------
        v : number or str
            Candidate color value

        allow_number : bool
            True if numbers are allowed as colors

        Returns
        -------
        number or str or None
        """
    if isinstance(v, numbers.Number) and allow_number:
        return v
    elif not isinstance(v, str):
        return None
    else:
        v_normalized = v.replace(' ', '').lower()
        if fullmatch(ColorValidator.re_hex, v_normalized):
            return v
        elif fullmatch(ColorValidator.re_rgb_etc, v_normalized):
            return v
        elif fullmatch(ColorValidator.re_ddk, v_normalized):
            return v
        elif v_normalized in ColorValidator.named_colors:
            return v
        else:
            return None