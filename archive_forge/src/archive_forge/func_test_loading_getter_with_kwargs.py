import uuid
from testtools import matchers
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_loading_getter_with_kwargs(self):
    called_opts = set()
    vals = {'a-bool': False, 'a-float': 99.99}

    def _getter(opt):
        called_opts.add(opt.name)
        return str(vals[opt.name])
    p = utils.MockLoader().load_from_options_getter(_getter, a_int=66, a_str='another')
    self.assertEqual(set(('a-bool', 'a-float')), called_opts)
    self.assertFalse(p['a_bool'])
    self.assertEqual(99.99, p['a_float'])
    self.assertEqual('another', p['a_str'])
    self.assertEqual(66, p['a_int'])