import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
from dockerpycreds.utils import find_executable  # type: ignore
from wandb.docker import auth, www_authenticate
from wandb.errors import Error
def auth_token(registry: str, repo: str) -> Dict[str, str]:
    """Make a request to the root of a v2 docker registry to get the auth url.

    Always returns a dictionary, if there's no token key we couldn't authenticate
    """
    auth_info = auth_config.resolve_authconfig(registry)
    if auth_info:
        normalized = {k.lower(): v for k, v in auth_info.items()}
        normalized_auth_info = (normalized.get('username'), normalized.get('password'))
    else:
        normalized_auth_info = None
    response = requests.get(f'https://{registry}/v2/', timeout=3)
    if response.headers.get('www-authenticate'):
        try:
            info: Dict = www_authenticate.parse(response.headers['www-authenticate'])
        except ValueError:
            info = {}
    else:
        log.error(f'Received {response} when attempting to authenticate with {registry}')
        info = {}
    if info.get('bearer'):
        res = requests.get(info['bearer']['realm'] + '?service={}&scope=repository:{}:pull'.format(info['bearer']['service'], repo), auth=normalized_auth_info, timeout=3)
        res.raise_for_status()
        result_json: Dict[str, str] = res.json()
        return result_json
    return {}