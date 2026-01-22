import inspect
from docutils import nodes
from docutils.parsers import rst
from docutils.parsers.rst import directives
from docutils.statemachine import ViewList
from sphinx.util import logging
from sphinx.util.nodes import nested_parse_with_titles
from stevedore import extension
def report_load_failure(mgr, ep, err):
    LOG.warning(u'Failed to load %s: %s' % (ep.module, err))