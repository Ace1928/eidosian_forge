from typing import Any, MutableMapping
import wandb
from wandb.sdk.lib import ipython
def snake_to_camel(self, string):
    camel = ''.join([i.title() for i in string.split('_')])
    return camel[0].lower() + camel[1:]