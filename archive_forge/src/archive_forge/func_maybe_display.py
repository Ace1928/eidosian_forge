import json
import logging
import os
import re
import shutil
import sys
from base64 import b64encode
from typing import Dict
import requests
from requests.compat import urljoin
import wandb
import wandb.util
from wandb.sdk.lib import filesystem
def maybe_display(self) -> bool:
    if not self.displayed and (self.path or wandb.run):
        display(self)
    return self.displayed