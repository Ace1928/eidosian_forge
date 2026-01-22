import os.path
import unittest
from Cython.TestUtils import TransformTest
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.ParseTreeTransforms import _calculate_pickle_checksums
from Cython.Compiler.Nodes import *
from Cython.Compiler import Main, Symtab, Options
def test_parallel_directives_cimports(self):
    self.run_pipeline(self.pipeline, self.import_code)
    parallel_directives = self.pipeline[0].parallel_directives
    self.assertEqual(parallel_directives, self.expected_directives_dict)