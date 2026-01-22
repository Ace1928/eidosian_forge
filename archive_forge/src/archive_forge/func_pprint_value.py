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
def pprint_value(self, value, print_unit=False):
    """Applies the applicable formatter to the value.

        Args:
            value: Dimension value to format

        Returns:
            Formatted dimension value
        """
    own_type = type(value) if self.type is None else self.type
    formatter = self.value_format if self.value_format else self.type_formatters.get(own_type)
    if formatter:
        if callable(formatter):
            formatted_value = formatter(value)
        elif isinstance(formatter, str):
            if isinstance(value, (dt.datetime, dt.date)):
                formatted_value = value.strftime(formatter)
            elif isinstance(value, np.datetime64):
                formatted_value = util.dt64_to_dt(value).strftime(formatter)
            elif re.findall('\\{(\\w+)\\}', formatter):
                formatted_value = formatter.format(value)
            else:
                formatted_value = formatter % value
    else:
        formatted_value = str(bytes_to_unicode(value))
    if print_unit and self.unit is not None:
        formatted_value = formatted_value + ' ' + bytes_to_unicode(self.unit)
    return formatted_value