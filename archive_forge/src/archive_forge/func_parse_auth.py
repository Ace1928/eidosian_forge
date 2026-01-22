import base64
import json
import logging
import os
import platform
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import dockerpycreds  # type: ignore
@classmethod
def parse_auth(cls, entries: Dict[str, Dict[str, Any]], raise_on_error: bool=False) -> Dict[str, Dict[str, Any]]:
    """Parse authentication entries.

        Arguments:
          entries:        Dict of authentication entries.
          raise_on_error: If set to true, an invalid format will raise
                          InvalidConfigFileError
        Returns:
          Authentication registry.
        """
    conf = {}
    for registry, entry in entries.items():
        if not isinstance(entry, dict):
            log.debug(f'Config entry for key {registry} is not auth config')
            if raise_on_error:
                raise InvalidConfigFileError(f'Invalid configuration for registry {registry}')
            return {}
        if 'identitytoken' in entry:
            log.debug(f'Found an IdentityToken entry for registry {registry}')
            conf[registry] = {'IdentityToken': entry['identitytoken']}
            continue
        if 'auth' not in entry:
            log.debug(f'Auth data for {registry} is absent. Client might be using a credentials store instead.')
            conf[registry] = {}
            continue
        username, password = decode_auth(entry['auth'])
        log.debug(f'Found entry (registry={repr(registry)}, username={repr(username)})')
        conf[registry] = {'username': username, 'password': password, 'email': entry.get('email'), 'serveraddress': registry}
    return conf