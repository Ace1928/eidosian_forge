import argparse
from unittest import mock
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import floatingips
def test_show_floatingip(self):
    list_value = [{'id': '84c4d37e-1f8b-45ce-897b-16ad7f49b0e9'}, {'id': 'f180cf4c-f886-4dd1-8c36-854d17fbefb5'}]
    get_value = {'id': '84c4d37e-1f8b-45ce-897b-16ad7f49b0e9'}
    show_floatingip, floatingip_manager = self.create_show_command(list_value, get_value)
    args = argparse.Namespace(id='84c4d37e-1f8b-45ce-897b-16ad7f49b0e9')
    expected = [('id',), ('84c4d37e-1f8b-45ce-897b-16ad7f49b0e9',)]
    ret = show_floatingip.get_data(args)
    self.assertEqual(ret, expected)
    floatingip_manager.get.assert_called_once_with('84c4d37e-1f8b-45ce-897b-16ad7f49b0e9')