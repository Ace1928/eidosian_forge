from __future__ import annotations
import html
import os
import posixpath
import re
from collections.abc import Iterable
from os import path
from typing import Any, cast
from docutils import nodes
from docutils.nodes import Node
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.environment.adapters.indexentries import IndexEntries
from sphinx.locale import get_translation
from sphinx.util import logging
from sphinx.util.nodes import NodeMatcher
from sphinx.util.osutil import canon_path, make_filename
from sphinx.util.template import SphinxRenderer
def build_qhp(self, outdir: str | os.PathLike[str], outname: str) -> None:
    logger.info(__('writing project file...'))
    tocdoc = self.env.get_and_resolve_doctree(self.config.master_doc, self, prune_toctrees=False)
    sections = []
    matcher = NodeMatcher(addnodes.compact_paragraph, toctree=True)
    for node in tocdoc.traverse(matcher):
        sections.extend(self.write_toc(node))
    for indexname, indexcls, content, collapse in self.domain_indices:
        item = section_template % {'title': indexcls.localname, 'ref': indexname + self.out_suffix}
        sections.append(' ' * 4 * 4 + item)
    sections = '\n'.join(sections)
    keywords = []
    index = IndexEntries(self.env).create_index(self, group_entries=False)
    for key, group in index:
        for title, (refs, subitems, key_) in group:
            keywords.extend(self.build_keywords(title, refs, subitems))
    keywords = '\n'.join(keywords)
    if self.config.qthelp_namespace:
        nspace = self.config.qthelp_namespace
    else:
        nspace = 'org.sphinx.%s.%s' % (outname, self.config.version)
    nspace = re.sub('[^a-zA-Z0-9.\\-]', '', nspace)
    nspace = re.sub('\\.+', '.', nspace).strip('.')
    nspace = nspace.lower()
    with open(path.join(outdir, outname + '.qhp'), 'w', encoding='utf-8') as f:
        body = render_file('project.qhp', outname=outname, title=self.config.html_title, version=self.config.version, project=self.config.project, namespace=nspace, master_doc=self.config.master_doc, sections=sections, keywords=keywords, files=self.get_project_files(outdir))
        f.write(body)
    homepage = 'qthelp://' + posixpath.join(nspace, 'doc', self.get_target_uri(self.config.master_doc))
    startpage = 'qthelp://' + posixpath.join(nspace, 'doc', 'index%s' % self.link_suffix)
    logger.info(__('writing collection project file...'))
    with open(path.join(outdir, outname + '.qhcp'), 'w', encoding='utf-8') as f:
        body = render_file('project.qhcp', outname=outname, title=self.config.html_short_title, homepage=homepage, startpage=startpage)
        f.write(body)