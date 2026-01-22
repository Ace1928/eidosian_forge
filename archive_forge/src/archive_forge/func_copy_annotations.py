import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def copy_annotations(src, dest):
    """
    Copy annotations from the tokens listed in src to the tokens in dest
    """
    assert len(src) == len(dest)
    for src_tok, dest_tok in zip(src, dest):
        dest_tok.annotation = src_tok.annotation