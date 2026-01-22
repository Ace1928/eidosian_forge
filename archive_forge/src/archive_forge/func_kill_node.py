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
def kill_node(self, node_id: Optional[str]=None, num: Optional[int]=None, rand: Optional[str]=None):
    self._queue.put(('kill_node', dict(node_id=node_id, num=num, rand=rand)))