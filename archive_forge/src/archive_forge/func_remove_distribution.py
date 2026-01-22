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
def remove_distribution(self, dist):
    """
        Remove a distribution from the finder. This will update internal
        information about who provides what.
        :param dist: The distribution to remove.
        """
    logger.debug('removing distribution %s', dist)
    name = dist.key
    del self.dists_by_name[name]
    del self.dists[name, dist.version]
    for p in dist.provides:
        name, version = parse_name_and_version(p)
        logger.debug('Remove from provided: %s, %s, %s', name, version, dist)
        s = self.provided[name]
        s.remove((version, dist))
        if not s:
            del self.provided[name]