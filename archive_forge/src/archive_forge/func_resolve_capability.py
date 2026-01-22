import platform
import six
from blessed.colorspace import CGA_COLORS, X11_COLORNAMES_TO_RGB
def resolve_capability(term, attr):
    """
    Resolve a raw terminal capability using :func:`tigetstr`.

    :arg Terminal term: :class:`~.Terminal` instance.
    :arg str attr: terminal capability name.
    :returns: string of the given terminal capability named by ``attr``,
       which may be empty (u'') if not found or not supported by the
       given :attr:`~.Terminal.kind`.
    :rtype: str
    """
    if not term.does_styling:
        return u''
    val = curses.tigetstr(term._sugar.get(attr, attr))
    return u'' if val is None else val.decode('latin1')