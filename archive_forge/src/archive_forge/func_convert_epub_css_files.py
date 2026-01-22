import html
from os import path
from typing import Any, Dict, List, NamedTuple, Set, Tuple
from sphinx import package_dir
from sphinx.application import Sphinx
from sphinx.builders import _epub_base
from sphinx.config import ENUM, Config
from sphinx.locale import __
from sphinx.util import logging, xmlname_checker
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.i18n import format_date
from sphinx.util.osutil import make_filename
def convert_epub_css_files(app: Sphinx, config: Config) -> None:
    """This converts string styled epub_css_files to tuple styled one."""
    epub_css_files: List[Tuple[str, Dict[str, Any]]] = []
    for entry in config.epub_css_files:
        if isinstance(entry, str):
            epub_css_files.append((entry, {}))
        else:
            try:
                filename, attrs = entry
                epub_css_files.append((filename, attrs))
            except Exception:
                logger.warning(__('invalid css_file: %r, ignored'), entry)
                continue
    config.epub_css_files = epub_css_files