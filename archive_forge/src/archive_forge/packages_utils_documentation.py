from rpy2 import rinterface
from warnings import warn
from collections import defaultdict

    :param symbol_mapping: as returned by `_map_symbols`
    :param conflicts: as returned by `_map_symbols`
    :param on_conflict: action to take if conflict
    :param msg_prefix: prefix for error message
    :param exception: exception to raise
    