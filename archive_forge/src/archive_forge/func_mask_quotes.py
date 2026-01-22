import sys
import re
def mask_quotes(input):
    """Mask the quoted strings so we skip braces inside quoted strings."""
    search_re = re.compile('(.*?)' + QUOTE_RE_STR)
    return [search_re.sub(quote_replace, line) for line in input]