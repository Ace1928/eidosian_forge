import base64
import binascii
import logging
import bcrypt
import webob
from oslo_config import cfg
from oslo_middleware import base
def parse_entry(entry):
    """Extrace the username and crypted password from a user auth file entry

    :param: entry: Line from auth user file to use for authentication
    :returns: a tuple of username and crypted password
    :raises: ConfigInvalid if the password is not in the supported bcrypt
    format
    """
    username, crypted_str = entry.split(':', maxsplit=1)
    crypted = crypted_str.encode('utf-8')
    if crypted[:4] not in (b'$2y$', b'$2a$', b'$2b$'):
        error_msg = 'Only bcrypt digested passwords are supported for %(username)s' % {'username': username}
        raise webob.exc.HTTPBadRequest(detail=error_msg)
    return (username, crypted)