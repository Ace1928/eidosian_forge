import os
import re
from math import ceil
from typing import Any, Dict, List, Optional, Tuple
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.locale import __
from sphinx.transforms import SphinxTransform
from sphinx.util import epoch_to_rfc1123, logging, requests, rfc1123_to_epoch, sha1
from sphinx.util.images import get_image_extension, guess_mimetype, parse_data_uri
from sphinx.util.osutil import ensuredir
def get_filename_for(filename: str, mimetype: str) -> str:
    basename = os.path.basename(filename)
    basename = re.sub(CRITICAL_PATH_CHAR_RE, '_', basename)
    return os.path.splitext(basename)[0] + get_image_extension(mimetype)