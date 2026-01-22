import builtins
import datetime as dt
import re
import weakref
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import partial
from itertools import chain
from operator import itemgetter
import numpy as np
import param
from . import util
from .accessors import Apply, Opts, Redim
from .options import Options, Store, cleanup_custom_options
from .pprint import PrettyPrinter
from .tree import AttrTree
from .util import bytes_to_unicode
@property
def pprint_label(self):
    """The pretty-printed label string for the Dimension"""
    unit = '' if self.unit is None else type(self.unit)(self.unit_format).format(unit=self.unit)
    return bytes_to_unicode(self.label) + bytes_to_unicode(unit)