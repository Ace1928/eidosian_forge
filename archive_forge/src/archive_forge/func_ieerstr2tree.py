import re
from nltk.metrics import accuracy as _accuracy
from nltk.tag.mapping import map_tag
from nltk.tag.util import str2tuple
from nltk.tree import Tree
def ieerstr2tree(s, chunk_types=['LOCATION', 'ORGANIZATION', 'PERSON', 'DURATION', 'DATE', 'CARDINAL', 'PERCENT', 'MONEY', 'MEASURE'], root_label='S'):
    """
    Return a chunk structure containing the chunked tagged text that is
    encoded in the given IEER style string.
    Convert a string of chunked tagged text in the IEER named
    entity format into a chunk structure.  Chunks are of several
    types, LOCATION, ORGANIZATION, PERSON, DURATION, DATE, CARDINAL,
    PERCENT, MONEY, and MEASURE.

    :rtype: Tree
    """
    m = _IEER_DOC_RE.match(s)
    if m:
        return {'text': _ieer_read_text(m.group('text'), root_label), 'docno': m.group('docno'), 'doctype': m.group('doctype'), 'date_time': m.group('date_time'), 'headline': _ieer_read_text(m.group('headline'), root_label)}
    else:
        return _ieer_read_text(s, root_label)