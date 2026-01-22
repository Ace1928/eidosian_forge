import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import util
def testAcceptableMimeType(self):
    valid_pairs = (('*', 'text/plain'), ('*/*', 'text/plain'), ('text/*', 'text/plain'), ('*/plain', 'text/plain'), ('text/plain', 'text/plain'))
    for accept, mime_type in valid_pairs:
        self.assertTrue(util.AcceptableMimeType([accept], mime_type))
    invalid_pairs = (('text/*', 'application/json'), ('text/plain', 'application/json'))
    for accept, mime_type in invalid_pairs:
        self.assertFalse(util.AcceptableMimeType([accept], mime_type))
    self.assertTrue(util.AcceptableMimeType(['application/json', '*/*'], 'text/plain'))
    self.assertFalse(util.AcceptableMimeType(['application/json', 'img/*'], 'text/plain'))