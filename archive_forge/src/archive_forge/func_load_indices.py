import os.path as _path
import csv as _csv
from netaddr.compat import _open_binary
from netaddr.core import Subscriber, Publisher
def load_indices():
    """Load OUI and IAB lookup indices into memory"""
    load_index(OUI_INDEX, _open_binary(__package__, 'oui.idx'))
    load_index(IAB_INDEX, _open_binary(__package__, 'iab.idx'))