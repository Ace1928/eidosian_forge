import copy
import json
import logging
import os
import re
import time
from functools import partial, reduce
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials as OAuthCredentials
from googleapiclient import discovery, errors
from ray._private.accelerators import TPUAcceleratorManager, tpu
from ray.autoscaler._private.gcp.node import MAX_POLLS, POLL_INTERVAL, GCPNodeType
from ray.autoscaler._private.util import check_legacy_fields
def tpu_accelerator_config_to_type(accelerator_config: dict) -> str:
    """Convert a provided accelerator_config to accelerator_type.

    Args:
        accelerator_config: A dictionary defining the spec of a
            TPU accelerator. The dictionary should consist of
            the keys 'type', indicating the TPU chip type, and
            'topology', indicating the topology of the TPU.

    Returns:
        A string, accelerator_type, e.g. "v4-8".

    """
    generation = accelerator_config['type'].lower()
    topology = accelerator_config['topology']
    chip_dimensions = [int(chip_count) for chip_count in topology.split('x')]
    num_chips = reduce(lambda x, y: x * y, chip_dimensions)
    num_cores = num_chips * 2
    return f'{generation}-{num_cores}'