import logging
from typing import Optional, Tuple
import concurrent.futures
import ray.dashboard.modules.log.log_utils as log_utils
import ray.dashboard.modules.log.log_consts as log_consts
import ray.dashboard.utils as dashboard_utils
import ray.dashboard.optional_utils as dashboard_optional_utils
from ray._private.ray_constants import env_integer
import asyncio
import grpc
import io
import os
from pathlib import Path
from ray.core.generated import reporter_pb2
from ray.core.generated import reporter_pb2_grpc
from ray._private.ray_constants import (
def find_end_offset_file(file: io.BufferedIOBase) -> int:
    """
    Find the offset of the end of a file without changing the file pointer.

    Args:
        file: File object

    Returns:
        Offset of the end of a file.
    """
    old_pos = file.tell()
    file.seek(0, io.SEEK_END)
    end = file.tell()
    file.seek(old_pos, io.SEEK_SET)
    return end