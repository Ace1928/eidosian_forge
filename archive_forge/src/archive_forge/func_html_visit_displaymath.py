from os import path
from typing import Any, Dict, cast
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.domains.math import MathDomain
from sphinx.environment import BuildEnvironment
from sphinx.errors import ExtensionError
from sphinx.locale import get_translation
from sphinx.util.math import get_node_equation_number
from sphinx.writers.html import HTMLTranslator
from sphinxcontrib.jsmath.version import __version__
def html_visit_displaymath(self: HTMLTranslator, node: nodes.math_block) -> None:
    if node['nowrap']:
        self.body.append(self.starttag(node, 'div', CLASS='math notranslate nohighlight'))
        self.body.append(self.encode(node.astext()))
        self.body.append('</div>')
        raise nodes.SkipNode
    for i, part in enumerate(node.astext().split('\n\n')):
        part = self.encode(part)
        if i == 0:
            if node['number']:
                number = get_node_equation_number(self, node)
                self.body.append('<span class="eqno">(%s)' % number)
                self.add_permalink_ref(node, _('Permalink to this equation'))
                self.body.append('</span>')
            self.body.append(self.starttag(node, 'div', CLASS='math notranslate nohighlight'))
        else:
            self.body.append('<div class="math">')
        if '&' in part or '\\\\' in part:
            self.body.append('\\begin{split}' + part + '\\end{split}')
        else:
            self.body.append(part)
        self.body.append('</div>\n')
    raise nodes.SkipNode