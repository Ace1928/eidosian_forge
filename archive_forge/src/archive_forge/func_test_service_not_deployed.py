import collections
import datetime
import itertools
import json
import os
import sys
from unittest import mock
import uuid
import eventlet
from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.common import short_id
from heat.common import timeutils
from heat.db import api as db_api
from heat.db import models
from heat.engine import attributes
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import clients
from heat.engine import constraints
from heat.engine import dependencies
from heat.engine import environment
from heat.engine import node_data
from heat.engine import plugin_manager
from heat.engine import properties
from heat.engine import resource
from heat.engine import resources
from heat.engine.resources.openstack.heat import none_resource
from heat.engine.resources.openstack.heat import test_resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import support
from heat.engine import template
from heat.engine import translation
from heat.objects import resource as resource_objects
from heat.objects import resource_data as resource_data_object
from heat.objects import resource_properties_data as rpd_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
import neutronclient.common.exceptions as neutron_exp
@mock.patch.object(clients.OpenStackClients, 'client_plugin')
def test_service_not_deployed(self, mock_client_plugin_method):
    """Test availability of resource when the service is not deployed.

        When the service is not deployed, resource is considered as
        unavailable.
        """
    mock_service_types, mock_client_plugin = self._mock_client_plugin(['test_type_un_deployed'], False)
    mock_client_plugin_method.return_value = mock_client_plugin
    self.assertFalse(generic_rsrc.ResourceWithDefaultClientName.is_service_available(context=mock.Mock())[0])
    mock_client_plugin_method.assert_called_once_with(generic_rsrc.ResourceWithDefaultClientName.default_client_name)
    mock_service_types.assert_called_once_with()
    mock_client_plugin.does_endpoint_exist.assert_called_once_with(service_type='test_type_un_deployed', service_name=generic_rsrc.ResourceWithDefaultClientName.default_client_name)