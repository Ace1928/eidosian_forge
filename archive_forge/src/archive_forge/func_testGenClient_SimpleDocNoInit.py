import os
import unittest
from apitools.gen import gen_client
from apitools.gen import test_utils
from __future__ import absolute_import
import pkgutil
def testGenClient_SimpleDocNoInit(self):
    with test_utils.TempDir() as tmp_dir_path:
        gen_client.main([gen_client.__file__, '--init-file', 'none', '--infile', GetTestDataPath('dns', 'dns_v1.json'), '--outdir', tmp_dir_path, '--overwrite', '--root_package', 'google.apis', 'client'])
        expected_files = set(['dns_v1_client.py', 'dns_v1_messages.py'])
        self.assertEquals(expected_files, set(os.listdir(tmp_dir_path)))