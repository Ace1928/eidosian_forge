import json
import os
import time
from wandb_gql import gql
import wandb
from wandb import util
from wandb.apis.internal import Api
from wandb.sdk import lib as wandb_lib
from wandb.sdk.data_types.utils import val_to_json
def read_h5(self, path, val=None):
    self.open_h5()
    if not self._h5:
        wandb.termerror('Reading tensors from summary requires h5py')
    else:
        return self._h5.get('summary/' + '.'.join(path), val)