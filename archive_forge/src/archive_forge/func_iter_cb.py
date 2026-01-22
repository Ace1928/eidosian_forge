import numpy
import uuid
from .. import h5, h5s, h5t, h5a, h5p
from . import base
from .base import phil, with_phil, Empty, is_empty_dataspace, product
from .datatype import Datatype
def iter_cb(name, *args):
    """ Callback to gather attribute names """
    attrlist.append(self._d(name))