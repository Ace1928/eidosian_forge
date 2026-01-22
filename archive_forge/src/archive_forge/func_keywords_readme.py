import re
from nltk.corpus.reader.api import *
from nltk.tokenize import *
def keywords_readme(self):
    """
        Return the list of words and constituents considered as clues of a
        comparison (from listOfkeywords.txt).
        """
    keywords = []
    with self.open('listOfkeywords.txt') as fp:
        raw_text = fp.read()
    for line in raw_text.split('\n'):
        if not line or line.startswith('//'):
            continue
        keywords.append(line.strip())
    return keywords