import functools
import os
import re
import tempfile
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView
def get_sentences(self, sent_segm):
    id = self.get_segm_id(sent_segm[0])
    segm = self.text_view.segm_dict[id]
    beg = self.get_sent_beg(sent_segm[0])
    end = self.get_sent_end(sent_segm[len(sent_segm) - 1])
    return segm[beg:end]