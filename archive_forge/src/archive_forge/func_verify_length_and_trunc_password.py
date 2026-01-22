import itertools
from oslo_log import log
import passlib.hash
import keystone.conf
from keystone import exception
from keystone.i18n import _
def verify_length_and_trunc_password(password):
    """Verify and truncate the provided password to the max_password_length.

    We also need to check that the configured password hashing algorithm does
    not silently truncate the password.  For example, passlib.hash.bcrypt does
    this:
    https://passlib.readthedocs.io/en/stable/lib/passlib.hash.bcrypt.html#security-issues

    """
    BCRYPT_MAX_LENGTH = 72
    if CONF.identity.password_hash_algorithm == 'bcrypt' and CONF.identity.max_password_length > BCRYPT_MAX_LENGTH:
        msg = 'Truncating password to algorithm specific maximum length %d characters.'
        LOG.warning(msg, BCRYPT_MAX_LENGTH)
        max_length = BCRYPT_MAX_LENGTH
    else:
        max_length = CONF.identity.max_password_length
    try:
        password_utf8 = password.encode('utf-8')
        if len(password_utf8) > max_length:
            if CONF.strict_password_check:
                raise exception.PasswordVerificationError(size=max_length)
            else:
                msg = 'Truncating user password to %d characters.'
                LOG.warning(msg, max_length)
                return password_utf8[:max_length]
        else:
            return password_utf8
    except AttributeError:
        raise exception.ValidationError(attribute='string', target='password')