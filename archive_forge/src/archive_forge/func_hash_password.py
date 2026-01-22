import itertools
from oslo_log import log
import passlib.hash
import keystone.conf
from keystone import exception
from keystone.i18n import _
def hash_password(password):
    """Hash a password. Harder."""
    params = {}
    password_utf8 = verify_length_and_trunc_password(password)
    conf_hasher = CONF.identity.password_hash_algorithm
    hasher = _HASHER_NAME_MAP.get(conf_hasher)
    if hasher is None:
        raise RuntimeError(_('Password Hash Algorithm %s not found') % CONF.identity.password_hash_algorithm)
    if CONF.identity.password_hash_rounds:
        params['rounds'] = CONF.identity.password_hash_rounds
    if hasher is passlib.hash.scrypt:
        if CONF.identity.scrypt_block_size:
            params['block_size'] = CONF.identity.scrypt_block_size
        if CONF.identity.scrypt_parallelism:
            params['parallelism'] = CONF.identity.scrypt_parallelism
        if CONF.identity.salt_bytesize:
            params['salt_size'] = CONF.identity.salt_bytesize
    if hasher is passlib.hash.pbkdf2_sha512:
        if CONF.identity.salt_bytesize:
            params['salt_size'] = CONF.identity.salt_bytesize
    return hasher.using(**params).hash(password_utf8)