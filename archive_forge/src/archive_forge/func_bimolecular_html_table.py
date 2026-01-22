import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
@deprecated(last_supported_version='0.5.7', will_be_missing_in='0.8.0', use_instead='chempy.printing.tables.BimolecularTable')
def bimolecular_html_table(self, *args, **kwargs):
    from .printing.tables import BimolecularTable
    return BimolecularTable.from_ReactionSystem(self)