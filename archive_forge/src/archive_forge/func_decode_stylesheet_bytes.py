from webencodings import UTF8, decode, lookup
from .parser import parse_stylesheet
def decode_stylesheet_bytes(css_bytes, protocol_encoding=None, environment_encoding=None):
    """Determine the character encoding of a CSS stylesheet and decode it.

    This is based on the presence of a :abbr:`BOM (Byte Order Mark)`,
    a ``@charset`` rule, and encoding meta-information.

    :type css_bytes: :obj:`bytes`
    :param css_bytes: A CSS byte string.
    :type protocol_encoding: :obj:`str`
    :param protocol_encoding:
        The encoding label, if any, defined by HTTP or equivalent protocol.
        (e.g. via the ``charset`` parameter of the ``Content-Type`` header.)
    :type environment_encoding: :class:`webencodings.Encoding`
    :param environment_encoding:
        The `environment encoding
        <https://www.w3.org/TR/css-syntax/#environment-encoding>`_, if any.
    :returns:
        A 2-tuple of a decoded Unicode string and the
        :class:`webencodings.Encoding` object that was used.

    """
    if protocol_encoding:
        fallback = lookup(protocol_encoding)
        if fallback:
            return decode(css_bytes, fallback)
    if css_bytes.startswith(b'@charset "'):
        end_quote = css_bytes.find(b'"', 10, 100)
        if end_quote != -1 and css_bytes.startswith(b'";', end_quote):
            fallback = lookup(css_bytes[10:end_quote].decode('latin1'))
            if fallback:
                if fallback.name in ('utf-16be', 'utf-16le'):
                    return decode(css_bytes, UTF8)
                return decode(css_bytes, fallback)
    if environment_encoding:
        return decode(css_bytes, environment_encoding)
    return decode(css_bytes, UTF8)