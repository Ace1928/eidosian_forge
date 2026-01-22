import re
from nltk.stem.api import StemmerI
def nltk_condition(stem):
    """
            This has been modified from the original Porter algorithm so
            that y->i is only done when y is preceded by a consonant,
            but not if the stem is only a single consonant, i.e.

               (*c and not c) Y -> I

            So 'happy' -> 'happi', but
               'enjoy' -> 'enjoy'  etc

            This is a much better rule. Formerly 'enjoy'->'enjoi' and
            'enjoyment'->'enjoy'. Step 1c is perhaps done too soon; but
            with this modification that no longer really matters.

            Also, the removal of the contains_vowel(z) condition means
            that 'spy', 'fly', 'try' ... stem to 'spi', 'fli', 'tri' and
            conflate with 'spied', 'tried', 'flies' ...
            """
    return len(stem) > 1 and self._is_consonant(stem, len(stem) - 1)