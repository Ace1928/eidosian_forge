from __future__ import unicode_literals
import re
from tensorboard._vendor import html5lib
from tensorboard._vendor.html5lib.filters.base import Filter
from tensorboard._vendor.html5lib.filters.sanitizer import allowed_protocols
from tensorboard._vendor.html5lib.serializer import HTMLSerializer
from tensorboard._vendor.bleach import callbacks as linkify_callbacks
from tensorboard._vendor.bleach.encoding import force_unicode
from tensorboard._vendor.bleach.utils import alphabetize_attributes
def handle_email_addresses(self, src_iter):
    """Handle email addresses in character tokens"""
    for token in src_iter:
        if token['type'] == 'Characters':
            text = token['data']
            new_tokens = []
            end = 0
            for match in self.email_re.finditer(text):
                if match.start() > end:
                    new_tokens.append({u'type': u'Characters', u'data': text[end:match.start()]})
                attrs = {(None, u'href'): u'mailto:%s' % match.group(0), u'_text': match.group(0)}
                attrs = self.apply_callbacks(attrs, True)
                if attrs is None:
                    new_tokens.append({u'type': u'Characters', u'data': match.group(0)})
                else:
                    _text = attrs.pop(u'_text', '')
                    attrs = alphabetize_attributes(attrs)
                    new_tokens.extend([{u'type': u'StartTag', u'name': u'a', u'data': attrs}, {u'type': u'Characters', u'data': force_unicode(_text)}, {u'type': u'EndTag', u'name': 'a'}])
                end = match.end()
            if new_tokens:
                if end < len(text):
                    new_tokens.append({u'type': u'Characters', u'data': text[end:]})
                for new_token in new_tokens:
                    yield new_token
                continue
        yield token