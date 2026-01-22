import copy
import json
import logging
import os
import subprocess
import sys
import time
from threading import RLock
from types import ModuleType
from typing import Any, Dict, Optional
import yaml
import ray
import ray._private.ray_constants as ray_constants
from ray.autoscaler._private.fake_multi_node.command_runner import (
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def create_node_with_resources_and_labels(self, node_config, tags, count, resources, labels):
    with self.lock:
        is_head = tags[TAG_RAY_NODE_KIND] == NODE_KIND_HEAD
        if is_head:
            next_id = FAKE_HEAD_NODE_ID
        else:
            next_id = self._next_hex_node_id()
        self._nodes[next_id] = {'tags': tags, 'node_spec': self._create_node_spec_with_resources(head=is_head, node_id=next_id, resources=resources)}
        self._update_docker_compose_config()
        self._save_node_state()