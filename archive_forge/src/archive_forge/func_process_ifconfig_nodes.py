from typing import Any, Dict, List
from docutils import nodes
from docutils.nodes import Node
import sphinx
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles
from sphinx.util.typing import OptionSpec
def process_ifconfig_nodes(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    ns = {confval.name: confval.value for confval in app.config}
    ns.update(app.config.__dict__.copy())
    ns['builder'] = app.builder.name
    for node in list(doctree.findall(ifconfig)):
        try:
            res = eval(node['expr'], ns)
        except Exception as err:
            from traceback import format_exception_only
            msg = ''.join(format_exception_only(err.__class__, err))
            newnode = doctree.reporter.error('Exception occurred in ifconfig expression: \n%s' % msg, base_node=node)
            node.replace_self(newnode)
        else:
            if not res:
                node.replace_self([])
            else:
                node.replace_self(node.children)