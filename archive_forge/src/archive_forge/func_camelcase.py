import re
def camelcase(string):
    """ Convert string into camel case.

    Args:
        string: String to convert.

    Returns:
        string: Camel case string.

    """
    string = re.sub('^[\\-_\\.]', '', str(string))
    if not string:
        return string
    return uplowcase(string[0], 'low') + re.sub('[\\-_\\.\\s]([a-z0-9])', lambda matched: uplowcase(matched.group(1), 'up'), string[1:])