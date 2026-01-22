from os import path
from sys import version_info as python_version
from sphinx import version_info as sphinx_version
from sphinx.locale import _
from sphinx.util.logging import getLogger
def extend_html_context(app, pagename, templatename, context, doctree):
    context['sphinx_version_info'] = sphinx_version