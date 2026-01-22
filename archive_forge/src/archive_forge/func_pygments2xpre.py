import re
from io import StringIO
def pygments2xpre(s, language='python'):
    """Return markup suitable for XPreformatted"""
    try:
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
    except ImportError:
        return s
    from pygments.lexers import get_lexer_by_name
    rconv = lambda x: x
    out = StringIO()
    l = get_lexer_by_name(language)
    h = HtmlFormatter()
    highlight(s, l, h, out)
    styles = [(cls, style.split(';')[0].split(':')[1].strip()) for cls, (style, ttype, level) in h.class2style.items() if cls and style and style.startswith('color:')]
    return rconv(_2xpre(out.getvalue(), styles))