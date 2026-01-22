import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
def format_measure_timestamp(measurements):
    return '\n'.join([str(m[0]) for m in measurements])