import urllib
from typing import Optional
from wandb_gql import gql
import wandb
from wandb import util
from wandb.apis import public
from wandb.apis.attrs import Attrs
from wandb.sdk.lib import ipython
Generate HTML containing an iframe displaying this sweep.