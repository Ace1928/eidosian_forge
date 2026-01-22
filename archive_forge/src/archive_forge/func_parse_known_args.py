import argparse
import copy
import json
import os
import re
import sys
import yaml
import wandb
from wandb import trigger
from wandb.util import add_import_hook, get_optional_module
def parse_known_args(self, *args, **kwargs):
    args, unknown = super().parse_known_args(*args, **kwargs)
    if self._callback:
        self._callback(args, unknown=unknown)
    return (args, unknown)