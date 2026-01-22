from collections import defaultdict
from functools import lru_cache
import boto3
from boto3.exceptions import ResourceNotExistsError
from boto3.resources.base import ServiceResource
from botocore.client import BaseClient
from botocore.config import Config
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.constants import BOTO_MAX_RETRIES
@lru_cache()
def resource_cache(name, region, max_retries=BOTO_MAX_RETRIES, **kwargs) -> ServiceResource:
    cli_logger.verbose('Creating AWS resource `{}` in `{}`', cf.bold(name), cf.bold(region))
    kwargs.setdefault('config', Config(retries={'max_attempts': max_retries}))
    return boto3.resource(name, region, **kwargs)