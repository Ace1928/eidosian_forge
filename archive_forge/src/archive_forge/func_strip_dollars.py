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
def strip_dollars(text):
    """
    Remove all dollar symbols from text

    Parameters
    ----------
    text : str
        Text to remove dollars from
    """
    return text.strip('$')