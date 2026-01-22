import os
import tempfile
import textwrap
from openstack.config import loader
from openstack import exceptions
from openstack.tests.unit.config import base
def test__load_yaml_json_file_nonexisting(self):
    tested_files = []
    fn = os.path.join('/fake', 'file.txt')
    tested_files.append(fn)
    path, result = loader.OpenStackConfig()._load_yaml_json_file(tested_files)
    self.assertEqual(None, path)