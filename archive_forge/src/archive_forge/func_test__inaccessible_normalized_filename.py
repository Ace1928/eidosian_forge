import os
import sys
from unicodedata import normalize
from .. import osutils
from ..osutils import pathjoin
from . import TestCase, TestCaseWithTransport, TestSkipped
def test__inaccessible_normalized_filename(self):
    inf = osutils._inaccessible_normalized_filename
    self.assertEqual(('ascii', True), inf('ascii'))
    self.assertEqual((a_circle_c, True), inf(a_circle_c))
    self.assertEqual((a_circle_c, False), inf(a_circle_d))
    self.assertEqual((a_dots_c, True), inf(a_dots_c))
    self.assertEqual((a_dots_c, False), inf(a_dots_d))
    self.assertEqual((z_umlat_c, True), inf(z_umlat_c))
    self.assertEqual((z_umlat_c, False), inf(z_umlat_d))
    self.assertEqual((squared_c, True), inf(squared_c))
    self.assertEqual((squared_c, True), inf(squared_d))
    self.assertEqual((quarter_c, True), inf(quarter_c))
    self.assertEqual((quarter_c, True), inf(quarter_d))