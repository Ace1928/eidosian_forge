import html
import os
import re
from os import path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple
from urllib.parse import quote
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.utils import smartquotes
from sphinx import addnodes
from sphinx.builders.html import BuildInfo, StandaloneHTMLBuilder
from sphinx.locale import __
from sphinx.util import logging, status_iterator
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.i18n import format_date
from sphinx.util.osutil import copyfile, ensuredir
def build_content(self) -> None:
    """Write the metainfo file content.opf It contains bibliographic data,
        a file list and the spine (the reading order).
        """
    logger.info(__('writing content.opf file...'))
    metadata = self.content_metadata()
    if not self.outdir.endswith(os.sep):
        self.outdir += os.sep
    olen = len(self.outdir)
    self.files: List[str] = []
    self.ignored_files = ['.buildinfo', 'mimetype', 'content.opf', 'toc.ncx', 'META-INF/container.xml', 'Thumbs.db', 'ehthumbs.db', '.DS_Store', 'nav.xhtml', self.config.epub_basename + '.epub'] + self.config.epub_exclude_files
    if not self.use_index:
        self.ignored_files.append('genindex' + self.out_suffix)
    for root, dirs, files in os.walk(self.outdir):
        dirs.sort()
        for fn in sorted(files):
            filename = path.join(root, fn)[olen:]
            if filename in self.ignored_files:
                continue
            ext = path.splitext(filename)[-1]
            if ext not in self.media_types:
                if ext not in ('.js', '.xml'):
                    logger.warning(__('unknown mimetype for %s, ignoring'), filename, type='epub', subtype='unknown_project_files')
                continue
            filename = filename.replace(os.sep, '/')
            item = ManifestItem(html.escape(quote(filename)), html.escape(self.make_id(filename)), html.escape(self.media_types[ext]))
            metadata['manifest_items'].append(item)
            self.files.append(filename)
    spinefiles = set()
    for refnode in self.refnodes:
        if '#' in refnode['refuri']:
            continue
        if refnode['refuri'] in self.ignored_files:
            continue
        spine = Spine(html.escape(self.make_id(refnode['refuri'])), True)
        metadata['spines'].append(spine)
        spinefiles.add(refnode['refuri'])
    for info in self.domain_indices:
        spine = Spine(html.escape(self.make_id(info[0] + self.out_suffix)), True)
        metadata['spines'].append(spine)
        spinefiles.add(info[0] + self.out_suffix)
    if self.use_index:
        spine = Spine(html.escape(self.make_id('genindex' + self.out_suffix)), True)
        metadata['spines'].append(spine)
        spinefiles.add('genindex' + self.out_suffix)
    for name in self.files:
        if name not in spinefiles and name.endswith(self.out_suffix):
            spine = Spine(html.escape(self.make_id(name)), False)
            metadata['spines'].append(spine)
    html_tmpl = None
    if self.config.epub_cover:
        image, html_tmpl = self.config.epub_cover
        image = image.replace(os.sep, '/')
        metadata['cover'] = html.escape(self.make_id(image))
        if html_tmpl:
            spine = Spine(html.escape(self.make_id(self.coverpage_name)), True)
            metadata['spines'].insert(0, spine)
            if self.coverpage_name not in self.files:
                ext = path.splitext(self.coverpage_name)[-1]
                self.files.append(self.coverpage_name)
                item = ManifestItem(html.escape(self.coverpage_name), html.escape(self.make_id(self.coverpage_name)), html.escape(self.media_types[ext]))
                metadata['manifest_items'].append(item)
            ctx = {'image': html.escape(image), 'title': self.config.project}
            self.handle_page(path.splitext(self.coverpage_name)[0], ctx, html_tmpl)
            spinefiles.add(self.coverpage_name)
    auto_add_cover = True
    auto_add_toc = True
    if self.config.epub_guide:
        for type, uri, title in self.config.epub_guide:
            file = uri.split('#')[0]
            if file not in self.files:
                self.files.append(file)
            if type == 'cover':
                auto_add_cover = False
            if type == 'toc':
                auto_add_toc = False
            metadata['guides'].append(Guide(html.escape(type), html.escape(title), html.escape(uri)))
    if auto_add_cover and html_tmpl:
        metadata['guides'].append(Guide('cover', self.guide_titles['cover'], html.escape(self.coverpage_name)))
    if auto_add_toc and self.refnodes:
        metadata['guides'].append(Guide('toc', self.guide_titles['toc'], html.escape(self.refnodes[0]['refuri'])))
    copy_asset_file(path.join(self.template_dir, 'content.opf_t'), self.outdir, metadata)