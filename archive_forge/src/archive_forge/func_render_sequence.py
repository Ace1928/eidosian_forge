from __future__ import unicode_literals
import six
import pybtex.io
from pybtex.plugin import Plugin
def render_sequence(self, rendered_list):
    """Render a sequence of rendered Text objects.
        The default implementation simply concatenates
        the strings in rendered_list.
        Override this method for non-string backends.
        """
    return ''.join(rendered_list)