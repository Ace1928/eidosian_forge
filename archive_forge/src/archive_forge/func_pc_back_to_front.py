import ast
import json
import sys
import urllib
from wandb_gql import gql
import wandb
from wandb.apis import public
from wandb.apis.attrs import Attrs
from wandb.apis.paginator import Paginator
from wandb.sdk.lib import ipython
def pc_back_to_front(self, name):
    if name is None:
        return None
    elif 'summary:' in name:
        name = name.replace('summary:', '')
        return self.panel_metrics_helper.FRONTEND_NAME_MAPPING_REVERSED.get(name, name)
    elif name in self.FRONTEND_NAME_MAPPING_REVERSED:
        return self.FRONTEND_NAME_MAPPING_REVERSED[name]
    elif name in self.FRONTEND_NAME_MAPPING:
        return name
    elif name.startswith('config:') and '.value' in name:
        return name.replace('config:', '').replace('.value', '')
    elif name.startswith('summary_metrics.'):
        return name.replace('summary_metrics.', '')
    return name