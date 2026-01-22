from __future__ import print_function
import sys
from docutils import nodes
from docutils.statemachine import ViewList
from sphinx.util.compat import Directive
from sphinx.util.nodes import nested_parse_with_titles

    A directive to collect all lexers/formatters/filters and generate
    autoclass directives for them.
    