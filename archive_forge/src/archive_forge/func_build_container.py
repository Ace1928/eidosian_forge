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
def build_container(self, outname: str='META-INF/container.xml') -> None:
    """Write the metainfo file META-INF/container.xml."""
    logger.info(__('writing META-INF/container.xml file...'))
    outdir = path.join(self.outdir, 'META-INF')
    ensuredir(outdir)
    copy_asset_file(path.join(self.template_dir, 'container.xml'), outdir)