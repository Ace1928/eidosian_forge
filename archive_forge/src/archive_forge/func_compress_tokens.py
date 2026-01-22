import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def compress_tokens(tokens):
    """
    Combine adjacent tokens when there is no HTML between the tokens, 
    and they share an annotation
    """
    result = [tokens[0]]
    for tok in tokens[1:]:
        if not result[-1].post_tags and (not tok.pre_tags) and (result[-1].annotation == tok.annotation):
            compress_merge_back(result, tok)
        else:
            result.append(tok)
    return result