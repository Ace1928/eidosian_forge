import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
def get_obs_local_part(value):
    """ obs-local-part = word *("." word)
    """
    obs_local_part = ObsLocalPart()
    last_non_ws_was_dot = False
    while value and (value[0] == '\\' or value[0] not in PHRASE_ENDS):
        if value[0] == '.':
            if last_non_ws_was_dot:
                obs_local_part.defects.append(errors.InvalidHeaderDefect("invalid repeated '.'"))
            obs_local_part.append(DOT)
            last_non_ws_was_dot = True
            value = value[1:]
            continue
        elif value[0] == '\\':
            obs_local_part.append(ValueTerminal(value[0], 'misplaced-special'))
            value = value[1:]
            obs_local_part.defects.append(errors.InvalidHeaderDefect("'\\' character outside of quoted-string/ccontent"))
            last_non_ws_was_dot = False
            continue
        if obs_local_part and obs_local_part[-1].token_type != 'dot':
            obs_local_part.defects.append(errors.InvalidHeaderDefect("missing '.' between words"))
        try:
            token, value = get_word(value)
            last_non_ws_was_dot = False
        except errors.HeaderParseError:
            if value[0] not in CFWS_LEADER:
                raise
            token, value = get_cfws(value)
        obs_local_part.append(token)
    if obs_local_part[0].token_type == 'dot' or (obs_local_part[0].token_type == 'cfws' and obs_local_part[1].token_type == 'dot'):
        obs_local_part.defects.append(errors.InvalidHeaderDefect("Invalid leading '.' in local part"))
    if obs_local_part[-1].token_type == 'dot' or (obs_local_part[-1].token_type == 'cfws' and obs_local_part[-2].token_type == 'dot'):
        obs_local_part.defects.append(errors.InvalidHeaderDefect("Invalid trailing '.' in local part"))
    if obs_local_part.defects:
        obs_local_part.token_type = 'invalid-obs-local-part'
    return (obs_local_part, value)