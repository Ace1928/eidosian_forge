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
def get_conversion_rule(self, node: nodes.image) -> Tuple[str, str]:
    for candidate in self.guess_mimetypes(node):
        for supported in self.app.builder.supported_image_types:
            rule = (candidate, supported)
            if rule in self.conversion_rules:
                return rule
    return None