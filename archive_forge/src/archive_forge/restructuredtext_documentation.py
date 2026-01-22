from os import path
from docutils import nodes
from docutils.core import publish_doctree
from sphinx.application import Sphinx
from sphinx.io import SphinxStandaloneReader
from sphinx.parsers import RSTParser
from sphinx.util.docutils import sphinx_domains
Parse a string as reStructuredText with Sphinx application.