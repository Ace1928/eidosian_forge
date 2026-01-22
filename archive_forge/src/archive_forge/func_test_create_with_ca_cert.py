import collections
import json
from unittest import mock
from heatclient import exc
from heatclient.v1 import stacks
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.common import policy
from heat.common import template_format
from heat.engine.clients.os import heat_plugin
from heat.engine import environment
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.heat import remote_stack
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.tests import common as tests_common
from heat.tests import utils
def test_create_with_ca_cert(self):
    ca_cert = '-----BEGIN CERTIFICATE----- A CA CERT -----END CERTIFICATE-----'
    rsrc = self._create_with_remote_credential(ca_cert=ca_cert)
    self.assertEqual(ca_cert, rsrc._cacert)
    self.assertEqual(ca_cert, rsrc.cacert)
    self.assertIn('/tmp/', rsrc._ssl_verify)