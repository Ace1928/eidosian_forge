import sys
import time
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from oslo_utils import uuidutils
from saharaclient.api import base
def created_at_sorted(objs, reverse=False):
    return sorted(objs, key=created_at_key, reverse=reverse)