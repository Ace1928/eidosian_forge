import enum
import os
import sys
from typing import Dict, Optional, Tuple
import click
import wandb
from wandb.errors import AuthenticationError, UsageError
from wandb.old.settings import Settings as OldSettings
from ..apis import InternalApi
from .internal.internal_api import Api
from .lib import apikey
from .wandb_settings import Settings, Source
def update_session(self, key: Optional[str], status: ApiKeyStatus=ApiKeyStatus.VALID) -> None:
    _logger = wandb.setup()._get_logger()
    login_settings = dict()
    if status == ApiKeyStatus.OFFLINE:
        login_settings = dict(mode='offline')
    elif status == ApiKeyStatus.DISABLED:
        login_settings = dict(mode='disabled')
    elif key:
        login_settings = dict(api_key=key)
    self._wl._settings._apply_login(login_settings, _logger=_logger)
    if not self._wl.settings._offline:
        self._wl._update_user_settings()