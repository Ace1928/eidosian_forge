import base64
import os
import re
import textwrap
import warnings
from urllib.parse import quote
from xml.etree.ElementTree import Element
import bleach
from defusedxml import ElementTree  # type:ignore[import-untyped]
from nbconvert.preprocessors.sanitize import _get_default_css_sanitizer
def path2url(path):
    """Turn a file path into a URL"""
    parts = path.split(os.path.sep)
    return '/'.join((quote(part) for part in parts))