import numpy as np
from .. import util
from ..element import Element
from ..ndmapping import NdMapping, item_check, sorted_context
from .dictionary import DictInterface
from .interface import DataError, Interface

        Splits a multi-interface Dataset into regular Datasets using
        regular tabular interfaces.
        