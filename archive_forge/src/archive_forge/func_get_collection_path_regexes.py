from __future__ import annotations
import collections.abc as c
import json
import os
import re
import typing as t
from ...encoding import (
from ...io import (
from ...util import (
from ...util_common import (
from ...config import (
from ...python_requirements import (
from ...target import (
from ...data import (
from ...pypi_proxy import (
from ...provisioning import (
from ...coverage_util import (
def get_collection_path_regexes() -> tuple[t.Optional[t.Pattern], t.Optional[t.Pattern]]:
    """Return a pair of regexes used for identifying and manipulating collection paths."""
    if data_context().content.collection:
        collection_search_re = re.compile('/%s/' % data_context().content.collection.directory)
        collection_sub_re = re.compile('^.*?/%s/' % data_context().content.collection.directory)
    else:
        collection_search_re = None
        collection_sub_re = None
    return (collection_search_re, collection_sub_re)