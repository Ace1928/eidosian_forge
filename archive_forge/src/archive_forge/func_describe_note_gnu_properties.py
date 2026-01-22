from .enums import (
from .constants import (
from ..common.utils import bytes2hex
def describe_note_gnu_properties(properties, machine):
    descriptions = []
    for prop in properties:
        t, d, sz = (prop.pr_type, prop.pr_data, prop.pr_datasz)
        if t == 'GNU_PROPERTY_STACK_SIZE':
            if type(d) is int:
                prop_desc = 'stack size: 0x%x' % d
            else:
                prop_desc = 'stack size: <corrupt length: 0x%x>' % sz
        elif t == 'GNU_PROPERTY_NO_COPY_ON_PROTECTED':
            if sz != 0:
                prop_desc = ' <corrupt length: 0x%x>' % sz
            else:
                prop_desc = 'no copy on protected'
        elif t == 'GNU_PROPERTY_X86_FEATURE_1_AND':
            if sz != 4:
                prop_desc = ' <corrupt length: 0x%x>' % sz
            else:
                prop_desc = describe_note_gnu_property_bitmap_and(_DESCR_NOTE_GNU_PROPERTY_X86_FEATURE_1_FLAGS, 'x86 feature', d)
        elif t == 'GNU_PROPERTY_X86_FEATURE_2_USED':
            if sz != 4:
                prop_desc = ' <corrupt length: 0x%x>' % sz
            else:
                prop_desc = describe_note_gnu_property_bitmap_and(_DESCR_NOTE_GNU_PROPERTY_X86_FEATURE_2_FLAGS, 'x86 feature used', d)
        elif t == 'GNU_PROPERTY_X86_ISA_1_NEEDED':
            if sz != 4:
                prop_desc = ' <corrupt length: 0x%x>' % sz
            else:
                prop_desc = describe_note_gnu_property_bitmap_and(_DESCR_NOTE_GNU_PROPERTY_X86_ISA_1_FLAGS, 'x86 ISA needed', d)
        elif t == 'GNU_PROPERTY_X86_ISA_1_USED':
            if sz != 4:
                prop_desc = ' <corrupt length: 0x%x>' % sz
            else:
                prop_desc = describe_note_gnu_property_bitmap_and(_DESCR_NOTE_GNU_PROPERTY_X86_ISA_1_FLAGS, 'x86 ISA used', d)
        elif t == 'GNU_PROPERTY_AARCH64_FEATURE_1_AND' and machine == 'EM_AARCH64':
            if sz != 4:
                prop_desc = ' <corrupt length: 0x%x>' % sz
            else:
                prop_desc = describe_note_gnu_property_bitmap_and(_DESCR_NOTE_GNU_PROPERTY_AARCH64_FEATURE_1_AND, 'aarch64 feature', d)
        elif _DESCR_NOTE_GNU_PROPERTY_TYPE_LOPROC <= t <= _DESCR_NOTE_GNU_PROPERTY_TYPE_HIPROC:
            prop_desc = '<processor-specific type 0x%x data: %s >' % (t, bytes2hex(d, sep=' '))
        elif _DESCR_NOTE_GNU_PROPERTY_TYPE_LOUSER <= t <= _DESCR_NOTE_GNU_PROPERTY_TYPE_HIUSER:
            prop_desc = '<application-specific type 0x%x data: %s >' % (t, bytes2hex(d, sep=' '))
        else:
            prop_desc = '<unknown type 0x%x data: %s >' % (t, bytes2hex(d, sep=' '))
        descriptions.append(prop_desc)
    return '\n        '.join(descriptions)