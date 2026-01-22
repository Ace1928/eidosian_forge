from __future__ import unicode_literals
import base64
import codecs
import contextlib
import hashlib
import logging
import os
import posixpath
import sys
import zipimport
from . import DistlibException, resources
from .compat import StringIO
from .version import get_scheme, UnsupportedVersionError
from .metadata import (Metadata, METADATA_FILENAME, WHEEL_METADATA_FILENAME,
from .util import (parse_requirement, cached_property, parse_name_and_version,
def get_required_dists(dists, dist):
    """Recursively generate a list of distributions from *dists* that are
    required by *dist*.

    :param dists: a list of distributions
    :param dist: a distribution, member of *dists* for which we are interested
                 in finding the dependencies.
    """
    if dist not in dists:
        raise DistlibException('given distribution %r is not a member of the list' % dist.name)
    graph = make_graph(dists)
    req = set()
    todo = graph.adjacency_list[dist]
    seen = set((t[0] for t in todo))
    while todo:
        d = todo.pop()[0]
        req.add(d)
        pred_list = graph.adjacency_list[d]
        for pred in pred_list:
            d = pred[0]
            if d not in req and d not in seen:
                seen.add(d)
                todo.append(pred)
    return req