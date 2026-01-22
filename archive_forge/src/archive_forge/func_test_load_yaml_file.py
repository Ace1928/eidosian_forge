import os.path
import tempfile
import yaml
from mistralclient import utils
from oslo_serialization import jsonutils
from oslotest import base
def test_load_yaml_file(self):
    with tempfile.NamedTemporaryFile() as f:
        f.write(ENV_YAML.encode('utf-8'))
        f.flush()
        file_path = os.path.abspath(f.name)
        self.assertDictEqual(ENV_DICT, utils.load_file(file_path))