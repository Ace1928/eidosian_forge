import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_join_segment_parameters_raw(self):
    join_segment_parameters_raw = urlutils.join_segment_parameters_raw
    self.assertEqual('/somedir/path', join_segment_parameters_raw('/somedir/path'))
    self.assertEqual('/somedir/path,rawdata', join_segment_parameters_raw('/somedir/path', 'rawdata'))
    self.assertRaises(urlutils.InvalidURLJoin, join_segment_parameters_raw, '/somedir/path', 'rawdata1,rawdata2,rawdata3')
    self.assertEqual('/somedir/path,bla,bar', join_segment_parameters_raw('/somedir/path', 'bla', 'bar'))
    self.assertEqual('/somedir,exist=some/path,bla,bar', join_segment_parameters_raw('/somedir,exist=some/path', 'bla', 'bar'))
    self.assertRaises(TypeError, join_segment_parameters_raw, '/somepath', 42)