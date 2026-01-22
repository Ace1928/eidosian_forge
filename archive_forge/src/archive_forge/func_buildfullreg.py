import re
import sys
import six
from six.moves.urllib import parse as urlparse
from routes.util import _url_quote as url_quote, _str_encode, as_unicode
def buildfullreg(self, clist, include_names=True):
    """Build the regexp by iterating through the routelist and
        replacing dicts with the appropriate regexp match"""
    regparts = []
    for part in self.routelist:
        if isinstance(part, dict):
            var = part['name']
            if var == 'controller':
                partmatch = '|'.join(map(re.escape, clist))
            elif part['type'] == ':':
                partmatch = self.reqs.get(var) or '[^/]+?'
            elif part['type'] == '.':
                partmatch = self.reqs.get(var) or '[^/.]+?'
            else:
                partmatch = self.reqs.get(var) or '.+?'
            if include_names:
                regpart = '(?P<%s>%s)' % (var, partmatch)
            else:
                regpart = '(?:%s)' % partmatch
            if part['type'] == '.':
                regparts.append('(?:\\.%s)??' % regpart)
            else:
                regparts.append(regpart)
        else:
            regparts.append(re.escape(part))
    regexp = ''.join(regparts) + '$'
    return regexp