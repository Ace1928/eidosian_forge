import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List
from ray.autoscaler.v2.instance_manager.instance_storage import (
from ray.autoscaler.v2.instance_manager.ray_installer import RayInstaller
from ray.core.generated.instance_manager_pb2 import Instance
ThreadedRayInstaller is responsible for install ray on new nodes.