import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def pack_characters(self, info):
    """Packs sequences of characters into strings."""
    items = []
    characters = []
    case_flags = NOCASE
    for s in self.items:
        if type(s) is Character and s.positive and (not s.zerowidth):
            if s.case_flags != case_flags:
                if s.case_flags or is_cased_i(info, s.value):
                    Sequence._flush_characters(info, characters, case_flags, items)
                    case_flags = s.case_flags
            characters.append(s.value)
        elif type(s) is String or type(s) is Literal:
            if s.case_flags != case_flags:
                if s.case_flags or any((is_cased_i(info, c) for c in characters)):
                    Sequence._flush_characters(info, characters, case_flags, items)
                    case_flags = s.case_flags
            characters.extend(s.characters)
        else:
            Sequence._flush_characters(info, characters, case_flags, items)
            items.append(s.pack_characters(info))
    Sequence._flush_characters(info, characters, case_flags, items)
    return make_sequence(items)