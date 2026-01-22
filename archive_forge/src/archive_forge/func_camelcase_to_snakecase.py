import itertools
import os
import re
def camelcase_to_snakecase(name):
    """Convert camel-case string to snake-case."""
    name = _uppercase_uppercase_re.sub('\\1_\\2', name)
    name = _lowercase_uppercase_re.sub('\\1_\\2', name)
    return name.lower()