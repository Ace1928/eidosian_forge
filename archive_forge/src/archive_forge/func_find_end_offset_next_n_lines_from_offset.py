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
def find_end_offset_next_n_lines_from_offset(file: io.BufferedIOBase, start_offset: int, n: int) -> int:
    """
    Find the offsets of next n lines from a start offset.

    Args:
        file: File object
        start_offset: Start offset to read from, inclusive.
        n: Number of lines to find.

    Returns:
        Offset of the end of the next n line (exclusive)
    """
    file.seek(start_offset)
    end_offset = None
    for _ in range(n):
        line = file.readline()
        if not line:
            break
        end_offset = file.tell()
    logger.debug(f'Found next {n} lines from {start_offset} offset')
    return end_offset if end_offset is not None else file.seek(0, io.SEEK_END)