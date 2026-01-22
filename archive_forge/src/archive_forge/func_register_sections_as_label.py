from typing import Any, Dict, cast
from docutils import nodes
from docutils.nodes import Node
import sphinx
from sphinx.application import Sphinx
from sphinx.domains.std import StandardDomain
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.nodes import clean_astext
def register_sections_as_label(app: Sphinx, document: Node) -> None:
    domain = cast(StandardDomain, app.env.get_domain('std'))
    for node in document.findall(nodes.section):
        if app.config.autosectionlabel_maxdepth and get_node_depth(node) >= app.config.autosectionlabel_maxdepth:
            continue
        labelid = node['ids'][0]
        docname = app.env.docname
        title = cast(nodes.title, node[0])
        ref_name = getattr(title, 'rawsource', title.astext())
        if app.config.autosectionlabel_prefix_document:
            name = nodes.fully_normalize_name(docname + ':' + ref_name)
        else:
            name = nodes.fully_normalize_name(ref_name)
        sectname = clean_astext(title)
        logger.debug(__('section "%s" gets labeled as "%s"'), ref_name, name, location=node, type='autosectionlabel', subtype=docname)
        if name in domain.labels:
            logger.warning(__('duplicate label %s, other instance in %s'), name, app.env.doc2path(domain.labels[name][0]), location=node, type='autosectionlabel', subtype=docname)
        domain.anonlabels[name] = (docname, labelid)
        domain.labels[name] = (docname, labelid, sectname)