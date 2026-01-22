import tarfile
import zipfile
from .. import export, filter_tree, tests
from . import fixtures
from .test_filters import _stack_1
def test_zip_export_content_filter_tree(self):
    self.make_tree()
    export.export(self.filter_tree, 'out.zip')
    zipf = zipfile.ZipFile('out.zip', 'r')
    self.assertEqual(b'HELLO WORLD', zipf.read('out/hello'))