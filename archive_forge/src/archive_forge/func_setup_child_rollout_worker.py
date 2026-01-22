from collections import deque
from http.server import HTTPServer, SimpleHTTPRequestHandler
import logging
import queue
from socketserver import ThreadingMixIn
import threading
import time
import traceback
from typing import List
import ray.cloudpickle as pickle
from ray.rllib.env.policy_client import (
from ray.rllib.offline.input_reader import InputReader
from ray.rllib.offline.io_context import IOContext
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.evaluation.sampler import SamplerInput
from ray.rllib.utils.typing import SampleBatchType
def setup_child_rollout_worker():
    nonlocal lock
    with lock:
        nonlocal child_rollout_worker
        nonlocal inference_thread
        if child_rollout_worker is None:
            child_rollout_worker, inference_thread = _create_embedded_rollout_worker(rollout_worker.creation_args(), report_data)
            child_rollout_worker.set_weights(rollout_worker.get_weights())