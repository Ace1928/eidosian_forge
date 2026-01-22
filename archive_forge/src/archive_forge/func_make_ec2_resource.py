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
def make_ec2_resource(region, max_retries, aws_credentials=None) -> ServiceResource:
    """Make client, retrying requests up to `max_retries`."""
    aws_credentials = aws_credentials or {}
    return resource_cache('ec2', region, max_retries, **aws_credentials)