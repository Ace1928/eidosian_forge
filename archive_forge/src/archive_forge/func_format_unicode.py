import re
import html
def format_unicode(s: str) -> str:
    """Converts a string in CIF text-format to unicode.  Any HTML tags
    contained in the string are removed.  HTML numeric character references
    are unescaped (i.e. converted to unicode).

    Parameters:

    s: string
        The CIF text string to convert

    Returns:

    u: string
        A unicode formatted string.
    """
    s = html.unescape(s)
    s = multiple_replace(s, subs_dict)
    tagclean = re.compile('<.*?>')
    return re.sub(tagclean, '', s)