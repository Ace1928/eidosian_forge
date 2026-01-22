import json
import logging
import os
from dataclasses import dataclass
from typing import List
import ray
from ray.train._internal.utils import get_address_and_port
from ray.train._internal.worker_group import WorkerGroup
from ray.train.backend import Backend, BackendConfig
from ray.util import PublicAPI
def get_url():
    address, port = get_address_and_port()
    return f'{address}:{port}'