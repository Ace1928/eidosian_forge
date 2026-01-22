import tarfile
import zipfile
from .. import export, filter_tree, tests
from . import fixtures
from .test_filters import _stack_1
def test_tar_export_content_filter_tree(self):
    self.make_tree()
    export.export(self.filter_tree, 'out.tgz')
    ball = tarfile.open('out.tgz', 'r:gz')
    self.assertEqual(b'HELLO WORLD', ball.extractfile('out/hello').read())