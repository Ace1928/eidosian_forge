import codecs
import gzip
import os
import six.moves.urllib.request as urllib_request
import tempfile
import unittest
from apitools.gen import util
from mock import patch
def testKeywords(self):
    names = util.Names([''])
    self.assertEqual('in_', names.CleanName('in'))