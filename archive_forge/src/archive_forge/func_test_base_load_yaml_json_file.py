import os
import tempfile
import textwrap
from openstack.config import loader
from openstack import exceptions
from openstack.tests.unit.config import base
def test_base_load_yaml_json_file(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        tested_files = []
        for key, value in FILES.items():
            fn = os.path.join(tmpdir, 'file.{ext}'.format(ext=key))
            with open(fn, 'w+') as fp:
                fp.write(value)
            tested_files.append(fn)
        path, result = loader.OpenStackConfig()._load_yaml_json_file(tested_files)
        self.assertEqual(tmpdir, os.path.dirname(path))