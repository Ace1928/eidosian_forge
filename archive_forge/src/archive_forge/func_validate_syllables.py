import re
import warnings
from string import punctuation
from nltk.tokenize.api import TokenizerI
from nltk.util import ngrams
def validate_syllables(self, syllable_list):
    """
        Ensures each syllable has at least one vowel.
        If the following syllable doesn't have vowel, add it to the current one.

        :param syllable_list: Single word or token broken up into syllables.
        :type syllable_list: list(str)
        :return: Single word or token broken up into syllables
                 (with added syllables if necessary)
        :rtype: list(str)
        """
    valid_syllables = []
    front = ''
    vowel_pattern = re.compile('|'.join(self.vowels))
    for i, syllable in enumerate(syllable_list):
        if syllable in punctuation:
            valid_syllables.append(syllable)
            continue
        if not vowel_pattern.search(syllable):
            if len(valid_syllables) == 0:
                front += syllable
            else:
                valid_syllables = valid_syllables[:-1] + [valid_syllables[-1] + syllable]
        elif len(valid_syllables) == 0:
            valid_syllables.append(front + syllable)
        else:
            valid_syllables.append(syllable)
    return valid_syllables