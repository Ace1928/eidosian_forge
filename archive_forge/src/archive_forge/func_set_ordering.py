import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
def set_ordering(self, tag, ordering):
    """Set dictionary preferences for a language.

        The Enchant library supports the use of multiple dictionary programs
        and multiple languages.  This method specifies which dictionaries
        the broker should prefer when dealing with a given language.  'tag'
        must be an appropriate language specification and 'ordering' is a
        string listing the dictionaries in order of preference.  For example
        a valid ordering might be "aspell,myspell,ispell".
        The value of 'tag' can also be set to "*" to set a default ordering
        for all languages for which one has not been set explicitly.
        """
    self._check_this()
    _e.broker_set_ordering(self._this, tag.encode(), ordering.encode())