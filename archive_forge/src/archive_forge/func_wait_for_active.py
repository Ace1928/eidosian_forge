import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def wait_for_active(status_f, res_id):
    success = utils.wait_for_status(status_f=lambda x: _Munch(status_f(x)), res_id=res_id, status_field=constants.PROVISIONING_STATUS, sleep_time=3)
    if not success:
        raise exceptions.OctaviaClientException(code='n/a', message='The resource did not successfully reach ACTIVE status.')