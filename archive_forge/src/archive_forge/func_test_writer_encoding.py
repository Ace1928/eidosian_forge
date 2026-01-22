import argparse
import codecs
import io
from unittest import mock
from cliff import app as application
from cliff import command as c_cmd
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils as test_utils
from cliff import utils
import sys
def test_writer_encoding(self):
    text = 'tést'
    text_utf8 = text.encode('utf-8')
    out = io.StringIO()
    writer = codecs.getwriter('utf-8')(out)
    self.assertRaises(TypeError, writer.write, text)
    out = io.StringIO()
    writer = codecs.getwriter('utf-8')(out)
    self.assertRaises(TypeError, writer.write, text_utf8)