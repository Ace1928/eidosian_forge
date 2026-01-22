import json
import os
import time
from wandb_gql import gql
import wandb
from wandb import util
from wandb.apis.internal import Api
from wandb.sdk import lib as wandb_lib
from wandb.sdk.data_types.utils import val_to_json
def write_h5(self, path, val):
    self.open_h5()
    if not self._h5:
        wandb.termerror('Storing tensors in summary requires h5py')
    else:
        try:
            del self._h5['summary/' + '.'.join(path)]
        except KeyError:
            pass
        self._h5['summary/' + '.'.join(path)] = val
        self._h5.flush()