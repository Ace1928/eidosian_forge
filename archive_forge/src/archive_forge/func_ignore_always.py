import array
import warnings
import enchant
from enchant.errors import (
from enchant.tokenize import get_tokenizer
from enchant.utils import get_default_language
def ignore_always(self, word=None):
    """Add given word to list of words to ignore.

        If no word is given, the current erroneous word is added.
        """
    if word is None:
        word = self.word
    word = self.coerce_string(word)
    if word not in self._ignore_words:
        self._ignore_words[word] = True