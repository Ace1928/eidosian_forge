import testtools
from unittest import mock
from troveclient.compat import auth
from troveclient.compat import exceptions
def case_iterating_services_in_catalog(self, scObj):
    service1 = scObj.catalog[scObj.root_key]['serviceCatalog'][0]
    scObj.catalog = dict()
    scObj.root_key = 'access'
    scObj.catalog[scObj.root_key] = dict()
    scObj.service_type = 'no_match'
    scObj.catalog[scObj.root_key]['serviceCatalog'] = [service1]
    self.assertRaises(exceptions.EndpointNotFound, scObj._url_for)
    scObj.service_type = 'database'
    scObj.service_name = 'no_match'
    self.assertRaises(exceptions.EndpointNotFound, scObj._url_for)
    scObj = auth.ServiceCatalog()
    scObj.catalog = dict()
    scObj.root_key = 'access'
    scObj.catalog[scObj.root_key] = dict()
    scObj.catalog[scObj.root_key]['serviceCatalog'] = []
    self.assertRaises(exceptions.EndpointNotFound, scObj._url_for, attr='test_attr', filter_value='test_attr_value')