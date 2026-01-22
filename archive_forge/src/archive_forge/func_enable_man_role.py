from the :func:`setup()` function.
import logging
import types
import docutils.nodes
import docutils.utils
from humanfriendly.deprecation import get_aliases
from humanfriendly.text import compact, dedent, format
from humanfriendly.usage import USAGE_MARKER, render_usage
def enable_man_role(app):
    """
    Enable the ``:man:`` role for linking to Debian Linux manual pages.

    :param app: The Sphinx application object.

    This function registers the :func:`man_role()` function to handle the
    ``:man:`` role.
    """
    app.add_role('man', man_role)