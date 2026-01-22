import html
from os import path
from typing import Any, Dict, List, Tuple, cast
from sphinx import package_dir
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.domains.changeset import ChangeSetDomain
from sphinx.locale import _, __
from sphinx.theming import HTMLThemeFactory
from sphinx.util import logging
from sphinx.util.console import bold  # type: ignore
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.osutil import ensuredir, os_path
def hl(no: int, line: str) -> str:
    line = '<a name="L%s"> </a>' % no + html.escape(line)
    for x in hltext:
        if x in line:
            line = '<span class="hl">%s</span>' % line
            break
    return line