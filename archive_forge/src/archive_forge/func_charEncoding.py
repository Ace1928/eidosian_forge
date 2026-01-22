import re
import string
import warnings
from bleach._vendor.html5lib import (  # noqa: E402 module level import not at top of file
from bleach._vendor.html5lib import (
from bleach._vendor.html5lib.constants import (  # noqa: E402 module level import not at top of file
from bleach._vendor.html5lib.constants import (
from bleach._vendor.html5lib.filters.base import (
from bleach._vendor.html5lib.filters.sanitizer import (
from bleach._vendor.html5lib.filters.sanitizer import (
from bleach._vendor.html5lib._inputstream import (
from bleach._vendor.html5lib.serializer import (
from bleach._vendor.html5lib._tokenizer import (
from bleach._vendor.html5lib._trie import (
@property
def charEncoding(self):
    return self._inner_stream.charEncoding