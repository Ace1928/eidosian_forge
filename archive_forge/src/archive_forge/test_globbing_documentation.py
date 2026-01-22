import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
tests that multiple mixed slashes are collapsed to single forward
        slashes and trailing mixed slashes are removed