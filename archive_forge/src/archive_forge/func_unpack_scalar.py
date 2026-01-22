import sys
from collections.abc import Iterable
from functools import lru_cache
import numpy as np
from packaging.version import Version
from .. import util
from ..element import Element
from ..ndmapping import NdMapping, item_check, sorted_context
from . import pandas
from .interface import Interface
from .util import cached
@classmethod
def unpack_scalar(cls, dataset, data):
    """
        Given a dataset object and data in the appropriate format for
        the interface, return a simple scalar.
        """
    if ibis4():
        count = data.count().execute()
    else:
        count = data[[]].count().execute()
    if len(data.columns) > 1 or count != 1:
        return data
    return data.execute().iat[0, 0]