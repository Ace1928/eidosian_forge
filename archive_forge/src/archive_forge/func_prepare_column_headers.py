import sys
import time
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from oslo_utils import uuidutils
from saharaclient.api import base
def prepare_column_headers(columns, remap=None):
    remap = remap if remap else {}
    new_columns = []
    for c in columns:
        for old, new in remap.items():
            c = c.replace(old, new)
        new_columns.append(c.replace('_', ' ').capitalize())
    return new_columns