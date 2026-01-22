import zipfile
import six
import logging
import uuid
import warnings
import posixpath as zip_path
import os.path
from collections import OrderedDict
from lxml import etree
import ebooklib
from ebooklib.utils import parse_string, parse_html_string, guess_type, get_pages_for_items
def set_identifier(self, uid):
    """
        Sets unique id for this epub

        :Args:
          - uid: Value of unique identifier for this book
        """
    self.uid = uid
    self.set_unique_metadata('DC', 'identifier', self.uid, {'id': self.IDENTIFIER_ID})