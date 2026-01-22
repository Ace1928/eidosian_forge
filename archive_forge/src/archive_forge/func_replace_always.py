import array
import warnings
import enchant
from enchant.errors import (
from enchant.tokenize import get_tokenizer
from enchant.utils import get_default_language
def replace_always(self, word, repl=None):
    """Always replace given word with given replacement.

        If a single argument is given, this is used to replace the
        current erroneous word.  If two arguments are given, that
        combination is added to the list for future use.
        """
    if repl is None:
        repl = word
        word = self.word
    repl = self.coerce_string(repl)
    word = self.coerce_string(word)
    self._replace_words[word] = repl
    if self.word == word:
        self.replace(repl)