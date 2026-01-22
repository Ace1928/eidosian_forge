import os
import sys
from unicodedata import normalize
from .. import osutils
from ..osutils import pathjoin
from . import TestCase, TestCaseWithTransport, TestSkipped
def test__accessible_normalized_filename(self):
    anf = osutils._accessible_normalized_filename
    self.assertEqual(('ascii', True), anf('ascii'))
    self.assertEqual((a_circle_c, True), anf(a_circle_c))
    self.assertEqual((a_circle_c, True), anf(a_circle_d))
    self.assertEqual((a_dots_c, True), anf(a_dots_c))
    self.assertEqual((a_dots_c, True), anf(a_dots_d))
    self.assertEqual((z_umlat_c, True), anf(z_umlat_c))
    self.assertEqual((z_umlat_c, True), anf(z_umlat_d))
    self.assertEqual((squared_c, True), anf(squared_c))
    self.assertEqual((squared_c, True), anf(squared_d))
    self.assertEqual((quarter_c, True), anf(quarter_c))
    self.assertEqual((quarter_c, True), anf(quarter_d))