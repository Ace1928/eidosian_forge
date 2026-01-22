import os
import unittest
from apitools.gen import gen_client
from apitools.gen import test_utils
from __future__ import absolute_import
import pkgutil
def testGenPipPackage_SimpleDoc(self):
    with test_utils.TempDir() as tmp_dir_path:
        gen_client.main([gen_client.__file__, '--infile', GetTestDataPath('dns', 'dns_v1.json'), '--outdir', tmp_dir_path, '--overwrite', '--root_package', 'google.apis', 'pip_package'])
        self.assertEquals(set(['apitools', 'setup.py']), set(os.listdir(tmp_dir_path)))