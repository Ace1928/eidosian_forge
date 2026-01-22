import html
import os
import posixpath
import re
import sys
import warnings
from datetime import datetime
from os import path
from typing import IO, Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Type
from urllib.parse import quote
import docutils.readers.doctree
from docutils import nodes
from docutils.core import Publisher
from docutils.frontend import OptionParser
from docutils.io import DocTreeInput, StringOutput
from docutils.nodes import Node
from docutils.utils import relative_path
from sphinx import __display_version__, package_dir
from sphinx import version_info as sphinx_version
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.config import ENUM, Config
from sphinx.deprecation import RemovedInSphinx70Warning, deprecated_alias
from sphinx.domains import Domain, Index, IndexEntry
from sphinx.environment import BuildEnvironment
from sphinx.environment.adapters.asset import ImageAdapter
from sphinx.environment.adapters.indexentries import IndexEntries
from sphinx.environment.adapters.toctree import TocTree
from sphinx.errors import ConfigError, ThemeError
from sphinx.highlighting import PygmentsBridge
from sphinx.locale import _, __
from sphinx.search import js_index
from sphinx.theming import HTMLThemeFactory
from sphinx.util import isurl, logging, md5, progress_message, status_iterator
from sphinx.util.docutils import new_document
from sphinx.util.fileutil import copy_asset
from sphinx.util.i18n import format_date
from sphinx.util.inventory import InventoryFile
from sphinx.util.matching import DOTFILES, Matcher, patmatch
from sphinx.util.osutil import copyfile, ensuredir, os_path, relative_uri
from sphinx.util.tags import Tags
from sphinx.writers.html import HTMLTranslator, HTMLWriter
from sphinx.writers.html5 import HTML5Translator
import sphinxcontrib.serializinghtml  # NOQA
import sphinx.builders.dirhtml  # NOQA
import sphinx.builders.singlehtml  # NOQA
def get_doc_context(self, docname: str, body: str, metatags: str) -> Dict[str, Any]:
    """Collect items for the template context of a page."""
    prev = next = None
    parents = []
    rellinks = self.globalcontext['rellinks'][:]
    related = self.relations.get(docname)
    titles = self.env.titles
    if related and related[2]:
        try:
            next = {'link': self.get_relative_uri(docname, related[2]), 'title': self.render_partial(titles[related[2]])['title']}
            rellinks.append((related[2], next['title'], 'N', _('next')))
        except KeyError:
            next = None
    if related and related[1]:
        try:
            prev = {'link': self.get_relative_uri(docname, related[1]), 'title': self.render_partial(titles[related[1]])['title']}
            rellinks.append((related[1], prev['title'], 'P', _('previous')))
        except KeyError:
            prev = None
    while related and related[0]:
        try:
            parents.append({'link': self.get_relative_uri(docname, related[0]), 'title': self.render_partial(titles[related[0]])['title']})
        except KeyError:
            pass
        related = self.relations.get(related[0])
    if parents:
        parents.pop()
    parents.reverse()
    title_node = self.env.longtitles.get(docname)
    title = self.render_partial(title_node)['title'] if title_node else ''
    source_suffix = self.env.doc2path(docname, False)[len(docname):]
    if self.config.html_copy_source:
        sourcename = docname + source_suffix
        if source_suffix != self.config.html_sourcelink_suffix:
            sourcename += self.config.html_sourcelink_suffix
    else:
        sourcename = ''
    meta = self.env.metadata.get(docname)
    self_toc = TocTree(self.env).get_toc_for(docname, self)
    toc = self.render_partial(self_toc)['fragment']
    return {'parents': parents, 'prev': prev, 'next': next, 'title': title, 'meta': meta, 'body': body, 'metatags': metatags, 'rellinks': rellinks, 'sourcename': sourcename, 'toc': toc, 'display_toc': self.env.toc_num_entries[docname] > 1, 'page_source_suffix': source_suffix}