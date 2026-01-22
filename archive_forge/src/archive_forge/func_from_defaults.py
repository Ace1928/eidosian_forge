from __future__ import unicode_literals
from .base import Style
from .from_dict import style_from_dict
@classmethod
def from_defaults(cls, style_dict=None, pygments_style_cls=pygments_DefaultStyle, include_extensions=True):
    """ Deprecated. """
    return style_from_pygments(style_cls=pygments_style_cls, style_dict=style_dict, include_defaults=include_extensions)