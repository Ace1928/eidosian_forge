from .base import Style, DEFAULT_ATTRS, ANSI_COLOR_NAMES
from .defaults import DEFAULT_STYLE_EXTENSIONS
from .utils import merge_attrs, split_token_in_parts
from six.moves import range
def style_from_dict(style_dict, include_defaults=True):
    """
    Create a ``Style`` instance from a dictionary or other mapping.

    The dictionary is equivalent to the ``Style.styles`` dictionary from
    pygments, with a few additions: it supports 'reverse' and 'blink'.

    Usage::

        style_from_dict({
            Token: '#ff0000 bold underline',
            Token.Title: 'blink',
            Token.SomethingElse: 'reverse',
        })

    :param include_defaults: Include the defaults (built-in) styling for
        selected text, etc...)
    """
    assert isinstance(style_dict, Mapping)
    if include_defaults:
        s2 = {}
        s2.update(DEFAULT_STYLE_EXTENSIONS)
        s2.update(style_dict)
        style_dict = s2
    token_to_attrs = {}
    for ttype, styledef in sorted(style_dict.items()):
        attrs = DEFAULT_ATTRS
        if 'noinherit' not in styledef:
            for i in range(1, len(ttype) + 1):
                try:
                    attrs = token_to_attrs[ttype[:-i]]
                except KeyError:
                    pass
                else:
                    break
        for part in styledef.split():
            if part == 'noinherit':
                pass
            elif part == 'bold':
                attrs = attrs._replace(bold=True)
            elif part == 'nobold':
                attrs = attrs._replace(bold=False)
            elif part == 'italic':
                attrs = attrs._replace(italic=True)
            elif part == 'noitalic':
                attrs = attrs._replace(italic=False)
            elif part == 'underline':
                attrs = attrs._replace(underline=True)
            elif part == 'nounderline':
                attrs = attrs._replace(underline=False)
            elif part == 'blink':
                attrs = attrs._replace(blink=True)
            elif part == 'noblink':
                attrs = attrs._replace(blink=False)
            elif part == 'reverse':
                attrs = attrs._replace(reverse=True)
            elif part == 'noreverse':
                attrs = attrs._replace(reverse=False)
            elif part in ('roman', 'sans', 'mono'):
                pass
            elif part.startswith('border:'):
                pass
            elif part.startswith('bg:'):
                attrs = attrs._replace(bgcolor=_colorformat(part[3:]))
            else:
                attrs = attrs._replace(color=_colorformat(part))
        token_to_attrs[ttype] = attrs
    return _StyleFromDict(token_to_attrs)