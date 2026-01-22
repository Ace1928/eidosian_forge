import re
import string
import warnings
from bleach._vendor.html5lib import (  # noqa: E402 module level import not at top of file
from bleach._vendor.html5lib import (
from bleach._vendor.html5lib.constants import (  # noqa: E402 module level import not at top of file
from bleach._vendor.html5lib.constants import (
from bleach._vendor.html5lib.filters.base import (
from bleach._vendor.html5lib.filters.sanitizer import (
from bleach._vendor.html5lib.filters.sanitizer import (
from bleach._vendor.html5lib._inputstream import (
from bleach._vendor.html5lib.serializer import (
from bleach._vendor.html5lib._tokenizer import (
from bleach._vendor.html5lib._trie import (
def match_entity(stream):
    """Returns first entity in stream or None if no entity exists

    Note: For Bleach purposes, entities must start with a "&" and end with a
    ";". This ignores ambiguous character entities that have no ";" at the end.

    :arg stream: the character stream

    :returns: the entity string without "&" or ";" if it's a valid character
        entity; ``None`` otherwise

    """
    if stream[0] != '&':
        raise ValueError('Stream should begin with "&"')
    stream = stream[1:]
    stream = list(stream)
    possible_entity = ''
    end_characters = '<&=;' + string.whitespace
    if stream and stream[0] == '#':
        possible_entity = '#'
        stream.pop(0)
        if stream and stream[0] in ('x', 'X'):
            allowed = '0123456789abcdefABCDEF'
            possible_entity += stream.pop(0)
        else:
            allowed = '0123456789'
        while stream and stream[0] not in end_characters:
            c = stream.pop(0)
            if c not in allowed:
                break
            possible_entity += c
        if possible_entity and stream and (stream[0] == ';'):
            return possible_entity
        return None
    while stream and stream[0] not in end_characters:
        c = stream.pop(0)
        possible_entity += c
        if not ENTITIES_TRIE.has_keys_with_prefix(possible_entity):
            return None
    if possible_entity and stream and (stream[0] == ';'):
        return possible_entity
    return None