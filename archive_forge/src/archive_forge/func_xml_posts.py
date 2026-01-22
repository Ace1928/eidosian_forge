import re
import textwrap
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.corpus.reader.xmldocs import *
from nltk.internals import ElementWrapper
from nltk.tag import map_tag
from nltk.util import LazyConcatenation
def xml_posts(self, fileids=None):
    if self._wrap_etree:
        return concat([XMLCorpusView(fileid, 'Session/Posts/Post', self._wrap_elt) for fileid in self.abspaths(fileids)])
    else:
        return concat([XMLCorpusView(fileid, 'Session/Posts/Post') for fileid in self.abspaths(fileids)])