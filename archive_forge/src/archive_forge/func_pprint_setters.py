from collections import namedtuple
import contextlib
from functools import cache, wraps
import inspect
from inspect import Signature, Parameter
import logging
from numbers import Number, Real
import re
import warnings
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .colors import BoundaryNorm
from .cm import ScalarMappable
from .path import Path
from .transforms import (BboxBase, Bbox, IdentityTransform, Transform, TransformedBbox,
def pprint_setters(self, prop=None, leadingspace=2):
    """
        If *prop* is *None*, return a list of strings of all settable
        properties and their valid values.

        If *prop* is not *None*, it is a valid property name and that
        property will be returned as a string of property : valid
        values.
        """
    if leadingspace:
        pad = ' ' * leadingspace
    else:
        pad = ''
    if prop is not None:
        accepts = self.get_valid_values(prop)
        return f'{pad}{prop}: {accepts}'
    lines = []
    for prop in sorted(self.get_setters()):
        accepts = self.get_valid_values(prop)
        name = self.aliased_name(prop)
        lines.append(f'{pad}{name}: {accepts}')
    return lines