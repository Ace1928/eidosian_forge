from os import path
from sys import version_info as python_version
from sphinx import version_info as sphinx_version
from sphinx.locale import _
from sphinx.util.logging import getLogger
def get_html_theme_path():
    """Return list of HTML theme paths."""
    cur_dir = path.abspath(path.dirname(path.dirname(__file__)))
    return cur_dir