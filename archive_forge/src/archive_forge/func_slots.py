import abc
import os
import typing
import warnings
import weakref
import rpy2.rinterface
import rpy2.rinterface_lib.callbacks
from rpy2.robjects import conversion
@property
def slots(self):
    """ Attributes of the underlying R object as a Python mapping.

        The attributes can accessed and assigned by name (as if they
        were in a Python `dict`)."""
    if self.__slots is None:
        self.__slots = RSlots(self)
    return self.__slots