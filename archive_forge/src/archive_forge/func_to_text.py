from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from dns import rdata
from dns.name import Name
from dns.tokenizer import Tokenizer
def to_text(self, origin=None, relativize=True, **kwargs):
    tokens = ['%d' % self._priority, self._target.choose_relativity(origin, relativize).to_text()] + self._params
    return ' '.join(tokens)