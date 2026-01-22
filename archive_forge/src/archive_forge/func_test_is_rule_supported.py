from unittest import mock
from neutron_lib.api.definitions import portbindings
from neutron_lib import constants
from neutron_lib.services.qos import base as qos_base
from neutron_lib.services.qos import constants as qos_consts
from neutron_lib.tests import _base
def test_is_rule_supported(self):
    self.assertTrue(_make_driver().is_rule_supported(_make_rule(rule_type=qos_consts.RULE_TYPE_MINIMUM_BANDWIDTH, params={qos_consts.MIN_KBPS: None, qos_consts.DIRECTION: constants.EGRESS_DIRECTION})))
    self.assertFalse(_make_driver().is_rule_supported(_make_rule(rule_type=qos_consts.RULE_TYPE_MINIMUM_BANDWIDTH, params={qos_consts.MIN_KBPS: None, qos_consts.DIRECTION: constants.INGRESS_DIRECTION})))