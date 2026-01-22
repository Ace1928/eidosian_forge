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
def toc_metadata(self, level: int, navpoints: List[NavPoint]) -> Dict[str, Any]:
    """Create a dictionary with all metadata for the toc.ncx file
        properly escaped.
        """
    metadata: Dict[str, Any] = {}
    metadata['uid'] = self.config.epub_uid
    metadata['title'] = html.escape(self.config.epub_title)
    metadata['level'] = level
    metadata['navpoints'] = navpoints
    return metadata