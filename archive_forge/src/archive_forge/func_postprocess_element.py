from __future__ import unicode_literals
import sys
import datetime
from decimal import Decimal
import os
import logging
import hashlib
import warnings
from . import __author__, __copyright__, __license__, __version__
def postprocess_element(elements, processed):
    """Fix unresolved references"""
    if elements in processed:
        return
    processed.append(elements)
    for k, v in elements.items():
        if isinstance(v, Struct):
            if v != elements:
                try:
                    postprocess_element(v, processed)
                except RuntimeError as e:
                    warnings.warn(unicode(e), RuntimeWarning)
            if v.refers_to:
                if isinstance(v.refers_to, dict):
                    extend_element(v, v.refers_to)
                    v.refers_to = None
                else:
                    elements[k] = v.refers_to
            if v.array:
                elements[k] = [v]
        if isinstance(v, list):
            for n in v:
                if isinstance(n, (Struct, list)):
                    postprocess_element(n, processed)