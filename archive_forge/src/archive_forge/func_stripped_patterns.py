import re
def stripped_patterns():
    """
  Remove named groups from patterns so we can join them together without
  a name collision.
  """
    regex = re.compile('\\?P<[\\w_]+>')
    return [regex.sub('', pattern) for pattern in PATTERNS]