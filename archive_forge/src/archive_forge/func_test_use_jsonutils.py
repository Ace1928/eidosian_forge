import testtools
from neutron_lib.hacking import checks
from neutron_lib.hacking import translation_checks as tc
from neutron_lib.tests import _base as base
def test_use_jsonutils(self):

    def __get_msg(fun):
        msg = 'N521: jsonutils.%(fun)s must be used instead of json.%(fun)s' % {'fun': fun}
        return [(0, msg)]
    for method in ('dump', 'dumps', 'load', 'loads'):
        self.assertEqual(__get_msg(method), list(checks.use_jsonutils('json.%s(' % method, './neutron/common/rpc.py')))
        self.assertEqual(0, len(list(checks.use_jsonutils('jsonx.%s(' % method, './neutron/common/rpc.py'))))
        self.assertEqual(0, len(list(checks.use_jsonutils('json.%sx(' % method, './neutron/common/rpc.py'))))
        self.assertEqual(0, len(list(checks.use_jsonutils('json.%s' % method, './neutron/plugins/ml2/drivers/openvswitch/agent/xenapi/etc/xapi.d/plugins/netwrap'))))