import re
import textwrap
from collections import defaultdict
from nltk.corpus.reader.xmldocs import XMLCorpusReader
def longid(self, shortid):
    """Returns longid of a VerbNet class

        Given a short VerbNet class identifier (eg '37.10'), map it
        to a long id (eg 'confess-37.10').  If ``shortid`` is already a
        long id, then return it as-is"""
    if self._LONGID_RE.match(shortid):
        return shortid
    elif not self._SHORTID_RE.match(shortid):
        raise ValueError('vnclass identifier %r not found' % shortid)
    try:
        return self._shortid_to_longid[shortid]
    except KeyError as e:
        raise ValueError('vnclass identifier %r not found' % shortid) from e