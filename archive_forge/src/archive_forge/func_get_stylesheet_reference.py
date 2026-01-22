import sys
import os
import os.path
import re
import itertools
import warnings
import unicodedata
from docutils import ApplicationError, DataError, __version_info__
from docutils import nodes
from docutils.nodes import unescape
import docutils.io
from docutils.utils.error_reporting import ErrorOutput, SafeString
def get_stylesheet_reference(settings, relative_to=None):
    """
    Retrieve a stylesheet reference from the settings object.

    Deprecated. Use get_stylesheet_list() instead to
    enable specification of multiple stylesheets as a comma-separated
    list.
    """
    if settings.stylesheet_path:
        assert not settings.stylesheet, 'stylesheet and stylesheet_path are mutually exclusive.'
        if relative_to == None:
            relative_to = settings._destination
        return relative_path(relative_to, settings.stylesheet_path)
    else:
        return settings.stylesheet