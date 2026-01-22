import base64
import binascii
import logging
import bcrypt
import webob
from oslo_config import cfg
from oslo_middleware import base
def auth_entry(entry, password):
    """Compare a password with a single user auth file entry

    :param: entry: Line from auth user file to use for authentication
    :param: password: Password encoded as bytes
    :returns: A dictionary of WSGI environment values to append to the request
    :raises: HTTPUnauthorized, if the entry doesn't match supplied password or
        if the entry is crypted with a method other than bcrypt
    """
    username, crypted = parse_entry(entry)
    if not bcrypt.checkpw(password, crypted):
        LOG.info('Password for %s does not match', username)
        raise webob.exc.HTTPUnauthorized()
    return {'HTTP_X_USER': username, 'HTTP_X_USER_NAME': username}