import testtools
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
def test_parse_args_to_dict_bad_type(self):
    _specs = ['--badtypearg', 'type=badtype', 'val1']
    ex = self.assertRaises(exceptions.CommandError, neutronV20.parse_args_to_dict, _specs)
    self.assertEqual('Invalid value_specs --badtypearg type=badtype val1: type badtype is not supported', str(ex))