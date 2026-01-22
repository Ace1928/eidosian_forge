import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def merge_insert(ins_chunks, doc):
    """ doc is the already-handled document (as a list of text chunks);
    here we add <ins>ins_chunks</ins> to the end of that.  """
    unbalanced_start, balanced, unbalanced_end = split_unbalanced(ins_chunks)
    doc.extend(unbalanced_start)
    if doc and (not doc[-1].endswith(' ')):
        doc[-1] += ' '
    doc.append('<ins>')
    if balanced and balanced[-1].endswith(' '):
        balanced[-1] = balanced[-1][:-1]
    doc.extend(balanced)
    doc.append('</ins> ')
    doc.extend(unbalanced_end)