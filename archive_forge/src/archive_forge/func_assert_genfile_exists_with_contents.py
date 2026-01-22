from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import shutil
import tempfile
import unittest
from gae_ext_runtime import ext_runtime
def assert_genfile_exists_with_contents(self, gen_files, filename, contents):
    for gen_file in gen_files:
        if gen_file.filename == filename:
            self.assertEqual(gen_file.contents, contents)
            break
    else:
        self.fail('filename {} not found in generated files {}'.format(filename, gen_files))