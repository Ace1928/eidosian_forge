import gzip
import os
import tempfile
from .... import tests
from ..exporter import (_get_output_stream, check_ref_format,
from . import FastimportFeature
def test_rewrite_invalid(self):
    self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'foo./bar')))
    self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/foo/')))
    self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/foo.')))
    self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'./foo')))
    self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'.refs/foo')))
    self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/foo..bar')))
    self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/foo?bar')))
    self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/foo.lock')))
    self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/v@{ation')))
    self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/foo\x08ar')))
    self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/foo\\bar')))
    self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/foo bar')))
    self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/foo\x10bar')))
    self.assertTrue(check_ref_format(sanitize_ref_name_for_git(b'heads/foo\x7fbar')))