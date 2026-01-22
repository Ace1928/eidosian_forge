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
def find_start_offset_last_n_lines_from_offset(file: io.BufferedIOBase, offset: int, n: int, block_size: int=BLOCK_SIZE) -> int:
    """
    Find the offset of the beginning of the line of the last X lines from an offset.

    Args:
        file: File object
        offset: Start offset from which to find last X lines, -1 means end of file.
            The offset is exclusive, i.e. data at the offset is not included
            in the result.
        n: Number of lines to find
        block_size: Block size to read from file

    Returns:
        Offset of the beginning of the line of the last X lines from a start offset.
    """
    logger.debug(f'Finding last {n} lines from {offset} offset')
    if offset == -1:
        offset = file.seek(0, io.SEEK_END)
    else:
        file.seek(offset, io.SEEK_SET)
    if n == 0:
        return offset
    nbytes_from_end = 0
    file.seek(max(0, offset - 1), os.SEEK_SET)
    if file.read(1) != b'\n':
        n -= 1
    lines_more = n
    read_offset = max(0, offset - block_size)
    prev_offset = offset
    while lines_more >= 0 and read_offset >= 0:
        file.seek(read_offset, 0)
        block_data = file.read(min(block_size, prev_offset - read_offset))
        num_lines = block_data.count(b'\n')
        if num_lines > lines_more:
            lines = block_data.split(b'\n', num_lines - lines_more)
            nbytes_from_end += len(lines[-1])
            break
        lines_more -= num_lines
        nbytes_from_end += len(block_data)
        if read_offset == 0:
            break
        prev_offset = read_offset
        read_offset = max(0, read_offset - block_size)
    offset_read_start = offset - nbytes_from_end
    assert offset_read_start >= 0, f'Read start offset({offset_read_start}) should be non-negative'
    return offset_read_start