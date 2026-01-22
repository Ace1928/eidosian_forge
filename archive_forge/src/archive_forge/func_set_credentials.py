import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
def set_credentials(self, name, host, user, scheme=None, password=None, port=None, path=None, verify_certificates=None, realm=None):
    """Set authentication credentials for a host.

        Any existing credentials with matching scheme, host, port and path
        will be deleted, regardless of name.

        Args:
          name: An arbitrary name to describe this set of credentials.
          host: Name of the host that accepts these credentials.
          user: The username portion of these credentials.
          scheme: The URL scheme (e.g. ssh, http) the credentials apply to.
          password: Password portion of these credentials.
          port: The IP port on the host that these credentials apply to.
          path: A filesystem path on the host that these credentials apply to.
          verify_certificates: On https, verify server certificates if True.
          realm: The http authentication realm (optional).
        """
    values = {'host': host, 'user': user}
    if password is not None:
        values['password'] = password
    if scheme is not None:
        values['scheme'] = scheme
    if port is not None:
        values['port'] = '%d' % port
    if path is not None:
        values['path'] = path
    if verify_certificates is not None:
        values['verify_certificates'] = str(verify_certificates)
    if realm is not None:
        values['realm'] = realm
    config = self._get_config()
    for section, existing_values in config.iteritems():
        for key in ('scheme', 'host', 'port', 'path', 'realm'):
            if existing_values.get(key) != values.get(key):
                break
        else:
            del config[section]
    config.update({name: values})
    self._save()