import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def split_trailing_whitespace(word):
    """
    This function takes a word, such as 'test

' and returns ('test','

')
    """
    stripped_length = len(word.rstrip())
    return (word[0:stripped_length], word[stripped_length:])