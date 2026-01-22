from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.manila import share_network
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_update_fail(self):
    net = self._create_network('share_network', self.rsrc_defn, self.stack)
    self.client.share_networks.remove_security_service.side_effect = Exception()
    props = self.tmpl['resources']['share_network']['properties'].copy()
    props['security_services'] = []
    update_template = net.t.freeze(properties=props)
    run = scheduler.TaskRunner(net.update, update_template)
    self.assertRaises(exception.ResourceFailure, run)