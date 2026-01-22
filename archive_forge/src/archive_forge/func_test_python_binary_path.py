import platform
import re
from io import StringIO
from .. import tests, version, workingtree
from .scenarios import load_tests_apply_scenarios
def test_python_binary_path(self):
    self.permit_source_tree_branch_repo()
    sio = StringIO()
    version.show_version(show_config=False, show_copyright=False, to_file=sio)
    out = sio.getvalue()
    m = re.search('Python interpreter: (.*) [0-9]', out)
    self.assertIsNot(m, None)
    self.assertPathExists(m.group(1))