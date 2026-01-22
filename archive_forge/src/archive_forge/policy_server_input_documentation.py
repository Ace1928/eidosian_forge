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
Returns metrics computed on a policy client rollout worker.