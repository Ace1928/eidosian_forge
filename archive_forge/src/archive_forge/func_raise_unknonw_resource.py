import operator
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient.tests.unit.osc.v2.networking_bgpvpn import fakes
def raise_unknonw_resource(resource_path, name_or_id):
    if str(count - 2) in name_or_id:
        raise Exception()