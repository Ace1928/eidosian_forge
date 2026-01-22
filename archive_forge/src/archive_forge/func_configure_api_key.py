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
def configure_api_key(self, key):
    if self._settings._notebook and (not self._settings.silent):
        wandb.termwarn("If you're specifying your api key in code, ensure this code is not shared publicly.\nConsider setting the WANDB_API_KEY environment variable, or running `wandb login` from the command line.")
    apikey.write_key(self._settings, key)
    self.update_session(key)
    self._key = key