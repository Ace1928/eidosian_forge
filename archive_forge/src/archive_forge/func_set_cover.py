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
def set_cover(self, file_name, content, create_page=True):
    """
        Set cover and create cover document if needed.

        :Args:
          - file_name: file name of the cover page
          - content: Content for the cover image
          - create_page: Should cover page be defined. Defined as bool value (optional). Default value is True.
        """
    c0 = EpubCover(file_name=file_name)
    c0.content = content
    self.add_item(c0)
    if create_page:
        c1 = EpubCoverHtml(image_name=file_name)
        self.add_item(c1)
    self.add_metadata(None, 'meta', '', OrderedDict([('name', 'cover'), ('content', 'cover-img')]))