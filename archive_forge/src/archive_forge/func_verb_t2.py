from the word (i.e. prefixes, suffixes and infixes). It was evaluated and
import re
from nltk.stem.api import StemmerI
def verb_t2(self, token):
    """
        stem the future prefixes and suffixes
        """
    if len(token) > 6:
        for s2 in self.pl_si2:
            if token.startswith(self.verb_pr2[0]) and token.endswith(s2):
                return token[2:-2]
        if token.startswith(self.verb_pr2[1]) and token.endswith(self.pl_si2[0]):
            return token[2:-2]
        if token.startswith(self.verb_pr2[1]) and token.endswith(self.pl_si2[2]):
            return token[2:-2]
    if len(token) > 5 and token.startswith(self.verb_pr2[0]) and token.endswith('ن'):
        return token[2:-1]
    if len(token) > 5 and token.startswith(self.verb_pr2[1]) and token.endswith('ن'):
        return token[2:-1]