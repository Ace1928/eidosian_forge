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
def fix_genindex(self, tree: List[Tuple[str, List[Tuple[str, Any]]]]) -> None:
    """Fix href attributes for genindex pages."""
    for _key, columns in tree:
        for _entryname, (links, subitems, _key) in columns:
            for i, (ismain, link) in enumerate(links):
                m = self.refuri_re.match(link)
                if m:
                    links[i] = (ismain, self.fix_fragment(m.group(1), m.group(2)))
            for _subentryname, subentrylinks in subitems:
                for i, (ismain, link) in enumerate(subentrylinks):
                    m = self.refuri_re.match(link)
                    if m:
                        subentrylinks[i] = (ismain, self.fix_fragment(m.group(1), m.group(2)))