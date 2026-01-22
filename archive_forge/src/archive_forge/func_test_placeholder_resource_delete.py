from unittest import mock
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import none_resource as none
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_placeholder_resource_delete(self):
    self._test_delete()