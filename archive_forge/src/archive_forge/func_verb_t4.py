from the word (i.e. prefixes, suffixes and infixes). It was evaluated and
import re
from nltk.stem.api import StemmerI
def verb_t4(self, token):
    """
        stem the present prefixes
        """
    if len(token) > 3:
        for pr1 in self.verb_suf1:
            if token.startswith(pr1):
                return token[1:]
        if token.startswith('ي'):
            return token[1:]