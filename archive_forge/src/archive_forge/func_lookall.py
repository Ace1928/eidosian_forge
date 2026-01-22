from __future__ import absolute_import, print_function, division
import locale
from itertools import islice
from collections import defaultdict
from petl.compat import numeric_types, text_type
from petl import config
from petl.util.base import Table
from petl.io.sources import MemorySource
from petl.io.html import tohtml
def lookall(table, **kwargs):
    """
    Format the entire table as text for inspection in an interactive session.

    N.B., this will load the entire table into memory.

    See also :func:`petl.util.vis.look` and :func:`petl.util.vis.see`.

    """
    kwargs['limit'] = None
    return look(table, **kwargs)