import re
import textwrap
from collections import defaultdict
from nltk.corpus.reader.xmldocs import XMLCorpusReader
def wordnetids(self, vnclass=None):
    """
        Return a list of all wordnet identifiers that appear in any
        class, or in ``classid`` if specified.
        """
    if vnclass is None:
        return sorted(self._wordnet_to_class.keys())
    else:
        if isinstance(vnclass, str):
            vnclass = self.vnclass(vnclass)
        return sum((member.get('wn', '').split() for member in vnclass.findall('MEMBERS/MEMBER')), [])