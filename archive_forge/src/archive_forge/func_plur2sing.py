from the word (i.e. prefixes, suffixes and infixes). It was evaluated and
import re
from nltk.stem.api import StemmerI
def plur2sing(self, token):
    """
        transform the word from the plural form to the singular form.
        """
    if len(token) > 4:
        for ps2 in self.pl_si2:
            if token.endswith(ps2):
                return token[:-2]
    if len(token) > 5:
        for ps3 in self.pl_si3:
            if token.endswith(ps3):
                return token[:-3]
    if len(token) > 3 and token.endswith('ات'):
        return token[:-2]
    if len(token) > 3 and token.startswith('ا') and (token[2] == 'ا'):
        return token[:2] + token[3:]
    if len(token) > 4 and token.startswith('ا') and (token[-2] == 'ا'):
        return token[1:-2] + token[-1]