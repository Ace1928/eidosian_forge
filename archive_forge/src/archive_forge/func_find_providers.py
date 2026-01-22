import gzip
from io import BytesIO
import json
import logging
import os
import posixpath
import re
import zlib
from . import DistlibException
from .compat import (urljoin, urlparse, urlunparse, url2pathname, pathname2url,
from .database import Distribution, DistributionPath, make_dist
from .metadata import Metadata, MetadataInvalidError
from .util import (cached_property, ensure_slash, split_filename, get_project_data,
from .version import get_scheme, UnsupportedVersionError
from .wheel import Wheel, is_compatible
def find_providers(self, reqt):
    """
        Find the distributions which can fulfill a requirement.

        :param reqt: The requirement.
         :type reqt: str
        :return: A set of distribution which can fulfill the requirement.
        """
    matcher = self.get_matcher(reqt)
    name = matcher.key
    result = set()
    provided = self.provided
    if name in provided:
        for version, provider in provided[name]:
            try:
                match = matcher.match(version)
            except UnsupportedVersionError:
                match = False
            if match:
                result.add(provider)
                break
    return result