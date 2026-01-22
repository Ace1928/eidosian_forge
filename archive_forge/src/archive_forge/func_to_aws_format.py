import copy
import logging
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List
import botocore
from boto3.resources.base import ServiceResource
import ray
import ray._private.ray_constants as ray_constants
from ray.autoscaler._private.aws.cloudwatch.cloudwatch_helper import (
from ray.autoscaler._private.aws.config import bootstrap_aws
from ray.autoscaler._private.aws.utils import (
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.constants import BOTO_CREATE_MAX_RETRIES, BOTO_MAX_RETRIES
from ray.autoscaler._private.log_timer import LogTimer
from ray.autoscaler.node_launch_exception import NodeLaunchException
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def to_aws_format(tags):
    """Convert the Ray node name tag to the AWS-specific 'Name' tag."""
    if TAG_RAY_NODE_NAME in tags:
        tags['Name'] = tags[TAG_RAY_NODE_NAME]
        del tags[TAG_RAY_NODE_NAME]
    return tags