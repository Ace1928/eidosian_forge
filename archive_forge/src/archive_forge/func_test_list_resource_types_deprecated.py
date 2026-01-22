from unittest import mock
from heat.common import exception
from heat.engine import environment
from heat.engine import resource as res
from heat.engine import service
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
@mock.patch.object(res.Resource, 'is_service_available')
def test_list_resource_types_deprecated(self, mock_is_service_available):
    mock_is_service_available.return_value = (True, None)
    resources = self.eng.list_resource_types(self.ctx, 'DEPRECATED')
    self.assertIn('OS::Aodh::Alarm', resources)