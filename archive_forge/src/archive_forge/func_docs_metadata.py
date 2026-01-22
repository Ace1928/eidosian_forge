import itertools
import os
import re
import sys
import textwrap
import types
from collections import OrderedDict, defaultdict
from itertools import zip_longest
from operator import itemgetter
from pprint import pprint
from nltk.corpus.reader import XMLCorpusReader, XMLCorpusView
from nltk.util import LazyConcatenation, LazyIteratorList, LazyMap
def docs_metadata(self, name=None):
    """
        Return an index of the annotated documents in Framenet.

        Details for a specific annotated document can be obtained using this
        class's doc() function and pass it the value of the 'ID' field.

        >>> from nltk.corpus import framenet as fn
        >>> len(fn.docs()) in (78, 107) # FN 1.5 and 1.7, resp.
        True
        >>> set([x.corpname for x in fn.docs_metadata()])>=set(['ANC', 'KBEval',                     'LUCorpus-v0.3', 'Miscellaneous', 'NTI', 'PropBank'])
        True

        :param name: A regular expression pattern used to search the
            file name of each annotated document. The document's
            file name contains the name of the corpus that the
            document is from, followed by two underscores "__"
            followed by the document name. So, for example, the
            file name "LUCorpus-v0.3__20000410_nyt-NEW.xml" is
            from the corpus named "LUCorpus-v0.3" and the
            document name is "20000410_nyt-NEW.xml".
        :type name: str
        :return: A list of selected (or all) annotated documents
        :rtype: list of dicts, where each dict object contains the following
                keys:

                - 'name'
                - 'ID'
                - 'corpid'
                - 'corpname'
                - 'description'
                - 'filename'
        """
    try:
        ftlist = PrettyList(self._fulltext_idx.values())
    except AttributeError:
        self._buildcorpusindex()
        ftlist = PrettyList(self._fulltext_idx.values())
    if name is None:
        return ftlist
    else:
        return PrettyList((x for x in ftlist if re.search(name, x['filename']) is not None))