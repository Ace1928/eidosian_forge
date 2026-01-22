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
def get_links_of_type(self, link_type):
    """
        Returns list of additional links of specific type.

        :Returns:
          As tuple returns list of links.
        """
    return (link for link in self.links if link.get('type', '') == link_type)