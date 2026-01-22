import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def htmldiff_tokens(html1_tokens, html2_tokens):
    """ Does a diff on the tokens themselves, returning a list of text
    chunks (not tokens).
    """
    s = InsensitiveSequenceMatcher(a=html1_tokens, b=html2_tokens)
    commands = s.get_opcodes()
    result = []
    for command, i1, i2, j1, j2 in commands:
        if command == 'equal':
            result.extend(expand_tokens(html2_tokens[j1:j2], equal=True))
            continue
        if command == 'insert' or command == 'replace':
            ins_tokens = expand_tokens(html2_tokens[j1:j2])
            merge_insert(ins_tokens, result)
        if command == 'delete' or command == 'replace':
            del_tokens = expand_tokens(html1_tokens[i1:i2])
            merge_delete(del_tokens, result)
    result = cleanup_delete(result)
    return result