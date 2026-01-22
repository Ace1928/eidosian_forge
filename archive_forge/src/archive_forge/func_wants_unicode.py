import array
import warnings
import enchant
from enchant.errors import (
from enchant.tokenize import get_tokenizer
from enchant.utils import get_default_language
def wants_unicode(self):
    """Check whether the checker wants unicode strings.

        This method will return True if the checker wants unicode strings
        as input, False if it wants normal strings.  It's important to
        provide the correct type of string to the checker.
        """
    if self._text.typecode == 'u':
        return True
    return False