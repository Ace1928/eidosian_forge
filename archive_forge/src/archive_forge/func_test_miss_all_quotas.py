from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import keystone as k_plugin
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_miss_all_quotas(self):
    my_quota = self.stack['my_quota']
    props = self.stack.t.t['resources']['my_quota']['properties'].copy()
    for key in valid_properties:
        if key in props:
            del props[key]
    my_quota.t = my_quota.t.freeze(properties=props)
    my_quota.reparse()
    msg = 'At least one of the following properties must be specified: healthmonitor, listener, loadbalancer, member, pool.'
    self.assertRaisesRegex(exception.PropertyUnspecifiedError, msg, my_quota.validate)