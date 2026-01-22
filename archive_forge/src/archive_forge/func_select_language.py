import sys
from kivy.core import core_select_lib
def select_language(self, language):
    """
        From the set of registered languages, select the first language
        for `language`.

        :Parameters:
            `language`: str
                Language identifier. Needs to be one of the options returned by
                list_languages(). Sets the language used for spell checking and
                word suggestions.
        """
    raise NotImplementedError('select_language() method not implemented by abstract spelling base class!')