from __future__ import absolute_import, print_function, division
from petl.compat import pickle, next
from petl.util.base import Table
from petl.io.sources import read_source_from_arg, write_source_from_arg
def teepickle(table, source=None, protocol=-1, write_header=True):
    """
    Return a table that writes rows to a pickle file as they are iterated
    over.

    """
    return TeePickleView(table, source=source, protocol=protocol, write_header=write_header)