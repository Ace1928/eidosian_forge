import configparser
import logging
import os
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse, urlunparse
import wandb
@property
def remote_url(self) -> Any:
    if self._remote_url:
        return self._remote_url
    if not self.remote:
        return None
    parsed = urlparse(self.remote.url)
    hostname = parsed.hostname
    if parsed.port is not None:
        hostname = f'{hostname}:{parsed.port}'
    if parsed.password is not None:
        return urlunparse(parsed._replace(netloc=f'{parsed.username}:@{hostname}'))
    return urlunparse(parsed._replace(netloc=hostname))