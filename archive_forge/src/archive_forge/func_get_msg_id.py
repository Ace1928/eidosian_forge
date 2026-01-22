import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
def get_msg_id(value):
    """msg-id = [CFWS] "<" id-left '@' id-right  ">" [CFWS]
       id-left = dot-atom-text / obs-id-left
       id-right = dot-atom-text / no-fold-literal / obs-id-right
       no-fold-literal = "[" *dtext "]"
    """
    msg_id = MsgID()
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        msg_id.append(token)
    if not value or value[0] != '<':
        raise errors.HeaderParseError("expected msg-id but found '{}'".format(value))
    msg_id.append(ValueTerminal('<', 'msg-id-start'))
    value = value[1:]
    try:
        token, value = get_dot_atom_text(value)
    except errors.HeaderParseError:
        try:
            token, value = get_obs_local_part(value)
            msg_id.defects.append(errors.ObsoleteHeaderDefect('obsolete id-left in msg-id'))
        except errors.HeaderParseError:
            raise errors.HeaderParseError("expected dot-atom-text or obs-id-left but found '{}'".format(value))
    msg_id.append(token)
    if not value or value[0] != '@':
        msg_id.defects.append(errors.InvalidHeaderDefect('msg-id with no id-right'))
        if value and value[0] == '>':
            msg_id.append(ValueTerminal('>', 'msg-id-end'))
            value = value[1:]
        return (msg_id, value)
    msg_id.append(ValueTerminal('@', 'address-at-symbol'))
    value = value[1:]
    try:
        token, value = get_dot_atom_text(value)
    except errors.HeaderParseError:
        try:
            token, value = get_no_fold_literal(value)
        except errors.HeaderParseError as e:
            try:
                token, value = get_domain(value)
                msg_id.defects.append(errors.ObsoleteHeaderDefect('obsolete id-right in msg-id'))
            except errors.HeaderParseError:
                raise errors.HeaderParseError("expected dot-atom-text, no-fold-literal or obs-id-right but found '{}'".format(value))
    msg_id.append(token)
    if value and value[0] == '>':
        value = value[1:]
    else:
        msg_id.defects.append(errors.InvalidHeaderDefect("missing trailing '>' on msg-id"))
    msg_id.append(ValueTerminal('>', 'msg-id-end'))
    if value and value[0] in CFWS_LEADER:
        token, value = get_cfws(value)
        msg_id.append(token)
    return (msg_id, value)