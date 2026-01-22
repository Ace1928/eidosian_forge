import os
import re
import warnings
from datetime import datetime, timezone
from os import path
from typing import TYPE_CHECKING, Callable, Generator, List, NamedTuple, Optional, Tuple, Union
import babel.dates
from babel.messages.mofile import write_mo
from babel.messages.pofile import read_po
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.errors import SphinxError
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.osutil import SEP, canon_path, relpath
@property
def catalogs(self) -> Generator[CatalogInfo, None, None]:
    for basedir, filename in self.pofiles:
        domain = canon_path(path.splitext(filename)[0])
        yield CatalogInfo(basedir, domain, self.encoding)