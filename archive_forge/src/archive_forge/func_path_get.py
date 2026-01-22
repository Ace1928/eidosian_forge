import os.path
import fixtures
from oslo_config import cfg
import testtools
def path_get(self, project_file=None):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_file:
        return os.path.join(root, project_file)
    else:
        return root