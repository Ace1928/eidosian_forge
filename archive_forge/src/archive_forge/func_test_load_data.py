import tempfile
from yaql.cli.cli_functions import load_data
from yaql.language import exceptions
from yaql.language import specs
from yaql.language import yaqltypes
import yaql.tests
def test_load_data(self):
    context = {}
    self.assertIsNone(load_data('/temporarydir/some_random_filename', context))
    self.assertEqual(context, {})
    with tempfile.NamedTemporaryFile() as f:
        f.write(b'This is not JSON')
        f.flush()
        self.assertIsNone(load_data(f.name, context))
        self.assertEqual(context, {})
    with tempfile.NamedTemporaryFile() as f:
        f.write(b'{"foo": "bar"}')
        f.flush()
        self.assertIsNone(load_data(f.name, context))
        self.assertEqual(context['$'], {'foo': 'bar'})