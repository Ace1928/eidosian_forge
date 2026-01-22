import configparser
import logging
import os
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse, urlunparse
import wandb
@property
def dirty(self) -> Any:
    if not self.repo:
        return False
    try:
        return self.repo.is_dirty()
    except GitCommandError:
        return False