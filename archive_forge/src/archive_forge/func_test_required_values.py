import uuid
from testtools import matchers
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_required_values(self):
    opts = [loading.Opt('a', required=False), loading.Opt('b', required=True)]
    Plugin, Loader = utils.create_plugin(opts=opts)
    lo = Loader()
    v = uuid.uuid4().hex
    p1 = lo.load_from_options(b=v)
    self.assertEqual(v, p1['b'])
    e = self.assertRaises(exceptions.MissingRequiredOptions, lo.load_from_options, a=v)
    self.assertEqual(1, len(e.options))
    for o in e.options:
        self.assertIsInstance(o, loading.Opt)
    self.assertEqual('b', e.options[0].name)