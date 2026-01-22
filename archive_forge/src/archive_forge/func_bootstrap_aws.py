import copy
import itertools
import json
import logging
import os
import time
from collections import Counter
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional, Set, Tuple
import boto3
import botocore
from packaging.version import Version
from ray.autoscaler._private.aws.cloudwatch.cloudwatch_helper import (
from ray.autoscaler._private.aws.utils import (
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.providers import _PROVIDER_PRETTY_NAMES
from ray.autoscaler._private.util import check_legacy_fields
from ray.autoscaler.tags import NODE_TYPE_LEGACY_HEAD, NODE_TYPE_LEGACY_WORKER
def bootstrap_aws(config):
    config = copy.deepcopy(config)
    check_legacy_fields(config)
    config['head_node'] = {}
    config = _configure_from_launch_template(config)
    config = _configure_from_network_interfaces(config)
    config = _configure_iam_role(config)
    config = _configure_key_pair(config)
    global_event_system.execute_callback(CreateClusterEvent.ssh_keypair_downloaded, {'ssh_key_path': config['auth']['ssh_private_key']})
    config = _configure_subnet(config)
    config = _configure_security_group(config)
    _check_ami(config)
    return config