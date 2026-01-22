import re
from typing import List, Optional, Tuple
from pip._vendor.packaging.tags import (
def version_info_to_nodot(version_info: Tuple[int, ...]) -> str:
    return ''.join(map(str, version_info[:2]))