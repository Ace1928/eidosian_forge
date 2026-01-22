import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
from dockerpycreds.utils import find_executable  # type: ignore
from wandb.docker import auth, www_authenticate
from wandb.errors import Error
def image_id_from_registry(image_name: str) -> Optional[str]:
    """Get the docker id from a public or private registry."""
    registry, repository, tag = parse(image_name)
    res = None
    try:
        token = auth_token(registry, repository).get('token')
        if registry == 'index.docker.io':
            registry = 'registry-1.docker.io'
        res = requests.head(f'https://{registry}/v2/{repository}/manifests/{tag}', headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.docker.distribution.manifest.v2+json'}, timeout=5)
        res.raise_for_status()
    except requests.RequestException:
        log.error(f'Received {res} when attempting to get digest for {image_name}')
        return None
    return '@'.join([registry + '/' + repository, res.headers['Docker-Content-Digest']])