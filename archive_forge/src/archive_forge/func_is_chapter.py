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
def is_chapter(self):
    """
        Returns if this document is chapter or not.

        :Returns:
          Returns book value.
        """
    return False