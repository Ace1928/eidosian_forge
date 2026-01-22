from ._constants import *
def fix_flags(src, flags):
    if isinstance(src, str):
        if flags & SRE_FLAG_LOCALE:
            raise ValueError('cannot use LOCALE flag with a str pattern')
        if not flags & SRE_FLAG_ASCII:
            flags |= SRE_FLAG_UNICODE
        elif flags & SRE_FLAG_UNICODE:
            raise ValueError('ASCII and UNICODE flags are incompatible')
    else:
        if flags & SRE_FLAG_UNICODE:
            raise ValueError('cannot use UNICODE flag with a bytes pattern')
        if flags & SRE_FLAG_LOCALE and flags & SRE_FLAG_ASCII:
            raise ValueError('ASCII and LOCALE flags are incompatible')
    return flags