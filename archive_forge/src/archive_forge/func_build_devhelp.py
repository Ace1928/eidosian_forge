from __future__ import annotations
import gzip
import os
import re
from os import path
from typing import Any
from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.environment.adapters.indexentries import IndexEntries
from sphinx.locale import get_translation
from sphinx.util import logging
from sphinx.util.nodes import NodeMatcher
from sphinx.util.osutil import make_filename
def build_devhelp(self, outdir: str | os.PathLike[str], outname: str) -> None:
    logger.info(__('dumping devhelp index...'))
    root = etree.Element('book', title=self.config.html_title, name=self.config.project, link='index.html', version=self.config.version)
    tree = etree.ElementTree(root)
    chapters = etree.SubElement(root, 'chapters')
    tocdoc = self.env.get_and_resolve_doctree(self.config.master_doc, self, prune_toctrees=False)

    def write_toc(node: nodes.Node, parent: etree.Element) -> None:
        if isinstance(node, addnodes.compact_paragraph) or isinstance(node, nodes.bullet_list):
            for subnode in node:
                write_toc(subnode, parent)
        elif isinstance(node, nodes.list_item):
            item = etree.SubElement(parent, 'sub')
            for subnode in node:
                write_toc(subnode, item)
        elif isinstance(node, nodes.reference):
            parent.attrib['link'] = node['refuri']
            parent.attrib['name'] = node.astext()
    matcher = NodeMatcher(addnodes.compact_paragraph, toctree=Any)
    for node in tocdoc.findall(matcher):
        write_toc(node, chapters)
    functions = etree.SubElement(root, 'functions')
    index = IndexEntries(self.env).create_index(self)

    def write_index(title: str, refs: list[Any], subitems: Any) -> None:
        if len(refs) == 0:
            pass
        elif len(refs) == 1:
            etree.SubElement(functions, 'function', name=title, link=refs[0][1])
        else:
            for i, ref in enumerate(refs):
                etree.SubElement(functions, 'function', name='[%d] %s' % (i, title), link=ref[1])
        if subitems:
            parent_title = re.sub('\\s*\\(.*\\)\\s*$', '', title)
            for subitem in subitems:
                write_index('%s %s' % (parent_title, subitem[0]), subitem[1], [])
    for key, group in index:
        for title, (refs, subitems, key) in group:
            write_index(title, refs, subitems)
    xmlfile = path.join(outdir, outname + '.devhelp.gz')
    with gzip.open(xmlfile, 'w') as f:
        tree.write(f, 'utf-8')