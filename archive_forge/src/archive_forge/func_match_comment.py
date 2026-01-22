import codecs
import re
from mako import exceptions
from mako import parsetree
from mako.pygen import adjust_whitespace
def match_comment(self):
    """matches the multiline version of a comment"""
    match = self.match('<%doc>(.*?)</%doc>', re.S)
    if match:
        self.append_node(parsetree.Comment, match.group(1))
        return True
    else:
        return False