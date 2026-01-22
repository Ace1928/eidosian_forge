import copy
import hashlib
import json
import logging
import os
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Union
import botocore
from ray.autoscaler._private.aws.utils import client_cache, resource_cache
from ray.autoscaler.tags import NODE_KIND_HEAD, TAG_RAY_CLUSTER_NAME, TAG_RAY_NODE_KIND
def update_from_config(self, is_head_node: bool) -> None:
    """Discovers and applies CloudWatch config updates as required.

        Args:
            is_head_node: whether this node is the head node.
        """
    for config_type in CloudwatchConfigType:
        if CloudwatchHelper.cloudwatch_config_exists(self.provider_config, config_type.value):
            self._update_cloudwatch_config(config_type.value, is_head_node)