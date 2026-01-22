import re
import unicodedata
from itertools import groupby
from typing import Any, Dict, List, Optional, Pattern, Tuple, cast
from sphinx.builders import Builder
from sphinx.domains.index import IndexDomain
from sphinx.environment import BuildEnvironment
from sphinx.errors import NoUri
from sphinx.locale import _, __
from sphinx.util import logging, split_into
Create the real index from the collected index entries.