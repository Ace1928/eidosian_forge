from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import secret
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_validate_content_type_without_payload(self):
    props = {'name': 'secret', 'payload_content_type': 'text/plain'}
    defn = rsrc_defn.ResourceDefinition('secret', 'OS::Barbican::Secret', props)
    res = self._create_resource(defn.name, defn, self.stack)
    msg = 'payload_content_type cannot be specified without payload.'
    self.assertRaisesRegex(exception.ResourcePropertyDependency, msg, res.validate)