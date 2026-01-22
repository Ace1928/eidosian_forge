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
def get_instance_type(node_config):
    if not node_config:
        return None
    if 'InstanceType' in node_config:
        return node_config['InstanceType']
    if 'machineType' in node_config:
        return node_config['machineType']
    if 'azure_arm_parameters' in node_config and 'vmSize' in node_config['azure_arm_parameters']:
        return node_config['azure_arm_parameters']['vmSize']
    return None