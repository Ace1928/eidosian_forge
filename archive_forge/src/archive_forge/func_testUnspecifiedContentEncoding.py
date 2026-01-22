import codecs
import gzip
import os
import six.moves.urllib.request as urllib_request
import tempfile
import unittest
from apitools.gen import util
from mock import patch
def testUnspecifiedContentEncoding(self):
    data = 'regular non-gzipped content'
    with patch.object(urllib_request, 'urlopen', return_value=MockRequestResponse(data, '')):
        self.assertEqual(data, util._GetURLContent('unused_url_parameter'))