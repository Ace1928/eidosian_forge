import configparser
import logging
import os
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse, urlunparse
import wandb
@property
def repo(self) -> Optional[Repo]:
    if not self._repo_initialized:
        self._repo = self._init_repo()
    return self._repo