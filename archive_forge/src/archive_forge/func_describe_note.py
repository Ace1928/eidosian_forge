from .enums import (
from .constants import (
from ..common.utils import bytes2hex
def describe_note(x, machine):
    n_desc = x['n_desc']
    desc = ''
    if x['n_type'] == 'NT_GNU_ABI_TAG':
        if x['n_name'] == 'Android':
            desc = '\n   description data: %s ' % bytes2hex(x['n_descdata'])
        else:
            desc = '\n    OS: %s, ABI: %d.%d.%d' % (_DESCR_NOTE_ABI_TAG_OS.get(n_desc['abi_os'], _unknown), n_desc['abi_major'], n_desc['abi_minor'], n_desc['abi_tiny'])
    elif x['n_type'] == 'NT_GNU_BUILD_ID':
        desc = '\n    Build ID: %s' % n_desc
    elif x['n_type'] == 'NT_GNU_GOLD_VERSION':
        desc = '\n    Version: %s' % n_desc
    elif x['n_type'] == 'NT_GNU_PROPERTY_TYPE_0':
        desc = '\n      Properties: ' + describe_note_gnu_properties(x['n_desc'], machine)
    else:
        desc = '\n      description data: {}'.format(bytes2hex(n_desc))
    if x['n_type'] == 'NT_GNU_ABI_TAG' and x['n_name'] == 'Android':
        note_type = 'NT_VERSION'
        note_type_desc = 'version'
    else:
        note_type = x['n_type'] if isinstance(x['n_type'], str) else 'Unknown note type:'
        note_type_desc = '0x%.8x' % x['n_type'] if isinstance(x['n_type'], int) else _DESCR_NOTE_N_TYPE.get(x['n_type'], _unknown)
    return '%s (%s)%s' % (note_type, note_type_desc, desc)