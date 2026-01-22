from collections import namedtuple
from . import compat
from . import exceptions
from . import misc
from . import normalizers
from . import uri
def idna_encoder(name):
    if any((ord(c) > 128 for c in name)):
        try:
            return idna.encode(name.lower(), strict=True, std3_rules=True)
        except idna.IDNAError:
            raise exceptions.InvalidAuthority(self.authority)
    return name