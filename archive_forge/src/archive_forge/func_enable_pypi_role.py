from the :func:`setup()` function.
import logging
import types
import docutils.nodes
import docutils.utils
from humanfriendly.deprecation import get_aliases
from humanfriendly.text import compact, dedent, format
from humanfriendly.usage import USAGE_MARKER, render_usage
def enable_pypi_role(app):
    """
    Enable the ``:pypi:`` role for linking to the Python Package Index.

    :param app: The Sphinx application object.

    This function registers the :func:`pypi_role()` function to handle the
    ``:pypi:`` role.
    """
    app.add_role('pypi', pypi_role)