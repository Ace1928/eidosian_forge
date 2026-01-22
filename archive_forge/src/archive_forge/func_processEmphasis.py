from __future__ import absolute_import, unicode_literals, division
import re
import sys
from commonmark import common
from commonmark.common import normalize_uri, unescape_string
from commonmark.node import Node
from commonmark.normalize_reference import normalize_reference
def processEmphasis(self, stack_bottom):
    openers_bottom = {'_': stack_bottom, '*': stack_bottom, "'": stack_bottom, '"': stack_bottom}
    odd_match = False
    use_delims = 0
    closer = self.delimiters
    while closer is not None and closer.get('previous') != stack_bottom:
        closer = closer.get('previous')
    while closer is not None:
        if not closer.get('can_close'):
            closer = closer.get('next')
        else:
            opener = closer.get('previous')
            opener_found = False
            closercc = closer.get('cc')
            while opener is not None and opener != stack_bottom and (opener != openers_bottom[closercc]):
                odd_match = (closer.get('can_open') or opener.get('can_close')) and closer['origdelims'] % 3 != 0 and ((opener['origdelims'] + closer['origdelims']) % 3 == 0)
                if opener.get('cc') == closercc and opener.get('can_open') and (not odd_match):
                    opener_found = True
                    break
                opener = opener.get('previous')
            old_closer = closer
            if closercc == '*' or closercc == '_':
                if not opener_found:
                    closer = closer.get('next')
                else:
                    use_delims = 2 if closer['numdelims'] >= 2 and opener['numdelims'] >= 2 else 1
                    opener_inl = opener.get('node')
                    closer_inl = closer.get('node')
                    opener['numdelims'] -= use_delims
                    closer['numdelims'] -= use_delims
                    opener_inl.literal = opener_inl.literal[:len(opener_inl.literal) - use_delims]
                    closer_inl.literal = closer_inl.literal[:len(closer_inl.literal) - use_delims]
                    if use_delims == 1:
                        emph = Node('emph', None)
                    else:
                        emph = Node('strong', None)
                    tmp = opener_inl.nxt
                    while tmp and tmp != closer_inl:
                        nxt = tmp.nxt
                        tmp.unlink()
                        emph.append_child(tmp)
                        tmp = nxt
                    opener_inl.insert_after(emph)
                    self.removeDelimitersBetween(opener, closer)
                    if opener['numdelims'] == 0:
                        opener_inl.unlink()
                        self.removeDelimiter(opener)
                    if closer['numdelims'] == 0:
                        closer_inl.unlink()
                        tempstack = closer['next']
                        self.removeDelimiter(closer)
                        closer = tempstack
            elif closercc == "'":
                closer['node'].literal = '’'
                if opener_found:
                    opener['node'].literal = '‘'
                closer = closer['next']
            elif closercc == '"':
                closer['node'].literal = '”'
                if opener_found:
                    opener['node'].literal = '“'
                closer = closer['next']
            if not opener_found and (not odd_match):
                openers_bottom[closercc] = old_closer['previous']
                if not old_closer['can_open']:
                    self.removeDelimiter(old_closer)
    while self.delimiters is not None and self.delimiters != stack_bottom:
        self.removeDelimiter(self.delimiters)