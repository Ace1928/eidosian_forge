import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from pyspark.util import inheritable_thread_target
from ray.util.spark.cluster_init import _start_ray_worker_nodes
def log_request(self, code='-', size='-'):
    pass