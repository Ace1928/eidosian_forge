from unittest import mock
from saharaclient.osc import utils
from saharaclient.tests.unit import base
def test_prepare_data(self):
    data = {'id': '123', 'name_of_res': 'name', 'description': 'descr'}
    fields = ['id', 'name_of_res', 'description']
    expected_data = {'Description': 'descr', 'Id': '123', 'Name of res': 'name'}
    self.assertEqual(expected_data, utils.prepare_data(data, fields))
    fields = ['id', 'name_of_res']
    expected_data = {'Id': '123', 'Name of res': 'name'}
    self.assertEqual(expected_data, utils.prepare_data(data, fields))
    fields = ['name_of_res']
    expected_data = {'Name of res': 'name'}
    self.assertEqual(expected_data, utils.prepare_data(data, fields))