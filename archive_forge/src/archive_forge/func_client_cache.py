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
def client_cache(name, region, max_retries=BOTO_MAX_RETRIES, **kwargs) -> BaseClient:
    try:
        return resource_cache(name, region, max_retries, **kwargs).meta.client
    except ResourceNotExistsError:
        cli_logger.verbose('Creating AWS client `{}` in `{}`', cf.bold(name), cf.bold(region))
        kwargs.setdefault('config', Config(retries={'max_attempts': max_retries}))
        return boto3.client(name, region, **kwargs)