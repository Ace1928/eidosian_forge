import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def cleanup_delete(chunks):
    """ Cleans up any DEL_START/DEL_END markers in the document, replacing
    them with <del></del>.  To do this while keeping the document
    valid, it may need to drop some tags (either start or end tags).

    It may also move the del into adjacent tags to try to move it to a
    similar location where it was originally located (e.g., moving a
    delete into preceding <div> tag, if the del looks like (DEL_START,
    'Text</div>', DEL_END)"""
    while 1:
        try:
            pre_delete, delete, post_delete = split_delete(chunks)
        except NoDeletes:
            break
        unbalanced_start, balanced, unbalanced_end = split_unbalanced(delete)
        locate_unbalanced_start(unbalanced_start, pre_delete, post_delete)
        locate_unbalanced_end(unbalanced_end, pre_delete, post_delete)
        doc = pre_delete
        if doc and (not doc[-1].endswith(' ')):
            doc[-1] += ' '
        doc.append('<del>')
        if balanced and balanced[-1].endswith(' '):
            balanced[-1] = balanced[-1][:-1]
        doc.extend(balanced)
        doc.append('</del> ')
        doc.extend(post_delete)
        chunks = doc
    return chunks