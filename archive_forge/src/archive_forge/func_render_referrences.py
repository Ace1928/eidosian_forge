from typing import Dict, Any
from textwrap import indent
from ._list import render_list
from ..core import BaseRenderer, BlockState
from ..util import strip_end
def render_referrences(self, state: BlockState):
    images = state.env['inline_images']
    for index, token in enumerate(images):
        attrs = token['attrs']
        alt = self.render_children(token, state)
        ident = self.INLINE_IMAGE_PREFIX + str(index)
        yield ('.. |' + ident + '| image:: ' + attrs['url'] + '\n   :alt: ' + alt)