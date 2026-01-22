from the :func:`setup()` function.
import logging
import types
import docutils.nodes
import docutils.utils
from humanfriendly.deprecation import get_aliases
from humanfriendly.text import compact, dedent, format
from humanfriendly.usage import USAGE_MARKER, render_usage
def enable_deprecation_notes(app):
    """
    Enable documenting backwards compatibility aliases using the autodoc_ extension.

    :param app: The Sphinx application object.

    This function connects the :func:`deprecation_note_callback()` function to
    ``autodoc-process-docstring`` events.

    .. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
    """
    app.connect('autodoc-process-docstring', deprecation_note_callback)