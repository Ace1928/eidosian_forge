import testtools
from unittest import mock
from troveclient.compat import auth
from troveclient.compat import exceptions
def case_ambiguous_endpoint(self, scObj):
    scObj.service_type = 'trove'
    scObj.service_name = 'test_service_name'

    def side_effect_func_service(key):
        if key == 'type':
            return 'trove'
        elif key == 'name':
            return 'test_service_name'
        return None
    mock1 = mock.Mock()
    mock1.side_effect = side_effect_func_service
    service1 = mock.Mock()
    service1.get = mock1
    endpoint2 = {'test_attr': 'test_attr_value'}
    service1.__getitem__ = mock.Mock(return_value=[endpoint2])
    scObj.catalog[scObj.root_key]['serviceCatalog'] = [service1]
    self.assertRaises(exceptions.AmbiguousEndpoints, scObj._url_for, attr='test_attr', filter_value='test_attr_value')