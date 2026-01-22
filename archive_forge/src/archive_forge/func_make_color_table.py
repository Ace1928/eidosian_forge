import os
import warnings
from IPython.utils.ipstruct import Struct
def make_color_table(in_class):
    """Build a set of color attributes in a class.

    Helper function for building the :class:`TermColors` and
    :class`InputTermColors`.
    """
    for name, value in color_templates:
        setattr(in_class, name, in_class._base % value)