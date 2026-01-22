import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def make_nonce(self):
    global random
    if not random:
        import random
    return ''.join((chr(random.randrange(256)) for _i in range(self.nonce_length)))