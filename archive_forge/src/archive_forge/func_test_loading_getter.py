import uuid
from testtools import matchers
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_loading_getter(self):
    called_opts = []
    vals = {'a-int': 44, 'a-bool': False, 'a-float': 99.99, 'a-str': 'value'}
    val = uuid.uuid4().hex

    def _getter(opt):
        called_opts.append(opt.name)
        return str(vals[opt.name])
    p = utils.MockLoader().load_from_options_getter(_getter, other=val)
    self.assertEqual(set(vals), set(called_opts))
    for k, v in vals.items():
        self.assertEqual(v, p[k.replace('-', '_')])
    self.assertEqual(val, p['other'])