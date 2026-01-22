import datetime
from unittest import mock
from oslo_serialization import jsonutils
import webob
import wsme
from glance.api import policy
from glance.api.v2 import metadef_namespaces as namespaces
from glance.api.v2 import metadef_objects as objects
from glance.api.v2 import metadef_properties as properties
from glance.api.v2 import metadef_resource_types as resource_types
from glance.api.v2 import metadef_tags as tags
import glance.gateway
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
def test_namespace_index_resource_type_delete_race(self):
    request = unit_test_utils.get_fake_request()
    filters = {'resource_types': [RESOURCE_TYPE1]}
    real_gmrtr = self.namespace_controller.gateway.get_metadef_resource_type_repo

    def race_delete(*a, **k):
        self.db.metadef_namespace_delete(request.context, NAMESPACE3)
        return real_gmrtr(*a, **k)
    with mock.patch.object(self.namespace_controller.gateway, 'get_metadef_resource_type_repo') as g:
        g.side_effect = race_delete
        output = self.namespace_controller.index(request, filters=filters)
    output = output.to_dict()
    self.assertEqual(2, len(output['namespaces']))
    actual = set([namespace.namespace for namespace in output['namespaces']])
    expected = set([NAMESPACE1, NAMESPACE3])
    self.assertEqual(expected, actual)
    self.assertEqual(1, len(output['namespaces'][0].resource_type_associations))
    self.assertEqual(wsme.types.Unset, output['namespaces'][1].resource_type_associations)