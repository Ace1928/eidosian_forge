from .err import OperationalError
from functools import partial
import hashlib
def scramble_native_password(password, message):
    """Scramble used for mysql_native_password"""
    if not password:
        return b''
    stage1 = sha1_new(password).digest()
    stage2 = sha1_new(stage1).digest()
    s = sha1_new()
    s.update(message[:SCRAMBLE_LENGTH])
    s.update(stage2)
    result = s.digest()
    return _my_crypt(result, stage1)