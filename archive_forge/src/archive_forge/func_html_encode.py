import codecs
import os
import pipes
import re
import subprocess
import tempfile
from humanfriendly.terminal import (
from coloredlogs.converter.colors import (
def html_encode(text):
    """
    Encode characters with a special meaning as HTML.

    :param text: The plain text (a string).
    :returns: The text converted to HTML (a string).
    """
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    return text