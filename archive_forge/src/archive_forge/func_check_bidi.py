from . import idnadata
import bisect
import unicodedata
import re
from typing import Union, Optional
from .intranges import intranges_contain
def check_bidi(label: str, check_ltr: bool=False) -> bool:
    bidi_label = False
    for idx, cp in enumerate(label, 1):
        direction = unicodedata.bidirectional(cp)
        if direction == '':
            raise IDNABidiError('Unknown directionality in label {} at position {}'.format(repr(label), idx))
        if direction in ['R', 'AL', 'AN']:
            bidi_label = True
    if not bidi_label and (not check_ltr):
        return True
    direction = unicodedata.bidirectional(label[0])
    if direction in ['R', 'AL']:
        rtl = True
    elif direction == 'L':
        rtl = False
    else:
        raise IDNABidiError('First codepoint in label {} must be directionality L, R or AL'.format(repr(label)))
    valid_ending = False
    number_type = None
    for idx, cp in enumerate(label, 1):
        direction = unicodedata.bidirectional(cp)
        if rtl:
            if not direction in ['R', 'AL', 'AN', 'EN', 'ES', 'CS', 'ET', 'ON', 'BN', 'NSM']:
                raise IDNABidiError('Invalid direction for codepoint at position {} in a right-to-left label'.format(idx))
            if direction in ['R', 'AL', 'EN', 'AN']:
                valid_ending = True
            elif direction != 'NSM':
                valid_ending = False
            if direction in ['AN', 'EN']:
                if not number_type:
                    number_type = direction
                elif number_type != direction:
                    raise IDNABidiError('Can not mix numeral types in a right-to-left label')
        else:
            if not direction in ['L', 'EN', 'ES', 'CS', 'ET', 'ON', 'BN', 'NSM']:
                raise IDNABidiError('Invalid direction for codepoint at position {} in a left-to-right label'.format(idx))
            if direction in ['L', 'EN']:
                valid_ending = True
            elif direction != 'NSM':
                valid_ending = False
    if not valid_ending:
        raise IDNABidiError('Label ends with illegal codepoint directionality')
    return True