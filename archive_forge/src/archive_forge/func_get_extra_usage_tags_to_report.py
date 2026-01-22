import json
import logging
import threading
import os
import platform
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set
import requests
import yaml
import ray
import ray._private.ray_constants as ray_constants
import ray._private.usage.usage_constants as usage_constant
from ray.experimental.internal_kv import _internal_kv_initialized, _internal_kv_put
from ray.core.generated import usage_pb2, gcs_pb2
def get_extra_usage_tags_to_report(gcs_client) -> Dict[str, str]:
    """Get the extra usage tags from env var and gcs kv store.

    The env var should be given this way; key=value;key=value.
    If parsing is failed, it will return the empty data.

    Returns:
        Extra usage tags as kv pairs.
    """
    extra_usage_tags = dict()
    extra_usage_tags_env_var = os.getenv('RAY_USAGE_STATS_EXTRA_TAGS', None)
    if extra_usage_tags_env_var:
        try:
            kvs = extra_usage_tags_env_var.strip(';').split(';')
            for kv in kvs:
                k, v = kv.split('=')
                extra_usage_tags[k] = v
        except Exception as e:
            logger.info(f'Failed to parse extra usage tags env var. Error: {e}')
    valid_tag_keys = [tag_key.lower() for tag_key in TagKey.keys()]
    try:
        keys = gcs_client.internal_kv_keys(usage_constant.EXTRA_USAGE_TAG_PREFIX.encode(), namespace=usage_constant.USAGE_STATS_NAMESPACE.encode())
        for key in keys:
            value = gcs_client.internal_kv_get(key, namespace=usage_constant.USAGE_STATS_NAMESPACE.encode())
            key = key.decode('utf-8')
            key = key[len(usage_constant.EXTRA_USAGE_TAG_PREFIX):]
            assert key in valid_tag_keys
            extra_usage_tags[key] = value.decode('utf-8')
    except Exception as e:
        logger.info(f'Failed to get extra usage tags from kv store {e}')
    return extra_usage_tags