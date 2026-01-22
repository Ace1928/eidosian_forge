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
def special_front_to_back(self, name):
    if name is None:
        return name
    name, *rest = name.split('.')
    rest = '.' + '.'.join(rest) if rest else ''
    if name.startswith('c::'):
        name = name[3:]
        return f'config:{name}.value{rest}'
    if name.startswith('s::'):
        name = name[3:] + rest
        return f'summary:{name}'
    name = name + rest
    if name in self.RUN_MAPPING:
        return 'run:' + self.RUN_MAPPING[name]
    if name in self.FRONTEND_NAME_MAPPING:
        return 'summary:' + self.FRONTEND_NAME_MAPPING[name]
    if name == 'Index':
        return name
    return 'summary:' + name