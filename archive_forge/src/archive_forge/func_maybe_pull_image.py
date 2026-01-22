import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, Optional
import yaml
import ray
from ray._private.dict import deep_update
from ray.autoscaler._private.fake_multi_node.node_provider import (
from ray.util.queue import Empty, Queue
def maybe_pull_image(self):
    if self._docker_image:
        try:
            images_str = subprocess.check_output(f'docker image inspect {self._docker_image}', shell=True)
            images = json.loads(images_str)
        except Exception as e:
            logger.error(f'Error inspecting image {self._docker_image}: {e}')
            return
        if not images:
            try:
                subprocess.check_output(f'docker pull {self._docker_image}', shell=True)
            except Exception as e:
                logger.error(f'Error pulling image {self._docker_image}: {e}')