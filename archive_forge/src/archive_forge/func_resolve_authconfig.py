import base64
import json
import logging
import os
import platform
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import dockerpycreds  # type: ignore
def resolve_authconfig(self, registry: Optional[str]=None) -> Optional[Dict[str, Any]]:
    """Return the authentication data for a specific registry.

        As with the Docker client, legacy entries in the config with full URLs are
        stripped down to hostnames before checking for a match. Returns None if no match
        was found.
        """
    if self.creds_store or self.cred_helpers:
        store_name = self.get_credential_store(registry)
        if store_name is not None:
            log.debug(f'Using credentials store {store_name!r}')
            cfg = self._resolve_authconfig_credstore(registry, store_name)
            if cfg is not None:
                return cfg
            log.debug('No entry in credstore - fetching from auth dict')
    registry = resolve_index_name(registry) if registry else INDEX_NAME
    log.debug(f'Looking for auth entry for {repr(registry)}')
    if registry in self.auths:
        log.debug(f'Found {repr(registry)}')
        return self.auths[registry]
    for key, conf in self.auths.items():
        if resolve_index_name(key) == registry:
            log.debug(f'Found {repr(key)}')
            return conf
    log.debug('No entry found')
    return None