from datetime import datetime
import json
import logging
import numpy as np
import os
from urllib.parse import urlparse
import time
from ray.air._internal.json import SafeFallbackEncoder
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.offline.io_context import IOContext
from ray.rllib.offline.output_writer import OutputWriter
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.compression import pack, compression_supported
from ray.rllib.utils.typing import FileType, SampleBatchType
from typing import Any, Dict, List
Initializes a JsonWriter instance.

        Args:
            path: a path/URI of the output directory to save files in.
            ioctx: current IO context object.
            max_file_size: max size of single files before rolling over.
            compress_columns: list of sample batch columns to compress.
        