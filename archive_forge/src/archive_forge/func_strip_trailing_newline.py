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
def strip_trailing_newline(text):
    """
    Strips a newline from the end of text.
    """
    if text.endswith('\n'):
        text = text[:-1]
    return text