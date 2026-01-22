from unittest import mock
import testtools
from heatclient.common import utils
from heatclient.v1 import resource_types
def test_list_types_with_filters(self):
    filters = {'name': 'OS::Keystone::*', 'version': '5.0.0', 'support_status': 'SUPPORTED'}
    manager = resource_types.ResourceTypeManager(None)
    with mock.patch.object(manager, '_list') as mock_list:
        mock_list.return_value = None
        manager.list(filters=filters)
        self.assertEqual(1, mock_list.call_count)
        url, param = mock_list.call_args[0]
        self.assertEqual('resource_types', param)
        base_url, query_params = utils.parse_query_url(url)
        self.assertEqual('/%s' % manager.KEY, base_url)
        filters_params = {}
        for item in filters:
            filters_params[item] = [filters[item]]
        self.assertEqual(filters_params, query_params)