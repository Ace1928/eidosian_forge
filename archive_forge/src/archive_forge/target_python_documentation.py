import sys
from typing import List, Optional, Set, Tuple
from pip._vendor.packaging.tags import Tag
from pip._internal.utils.compatibility_tags import get_supported, version_info_to_nodot
from pip._internal.utils.misc import normalize_version_info
Exactly the same as get_sorted_tags, but returns a set.

        This is important for performance.
        