import collections
import time
from openstack.cloud import meta
from openstack import exceptions
def stop_check_no_action(a):
    return a.endswith('_COMPLETE') or a.endswith('_FAILED')