import configparser
import logging
import os
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse, urlunparse
import wandb
@property
def last_commit(self) -> Any:
    if self._commit:
        return self._commit
    if not self.repo:
        return None
    if not self.repo.head or not self.repo.head.is_valid():
        return None
    try:
        if len(self.repo.refs) > 0:
            return self.repo.head.commit.hexsha
        else:
            return self.repo.git.show_ref('--head').split(' ')[0]
    except Exception:
        logger.exception('Unable to find most recent commit in git')
        return None