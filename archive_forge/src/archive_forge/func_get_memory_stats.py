import abc
import os
import re
from googlecloudsdk.core.util import files
def get_memory_stats(self) -> (int, int):
    """Fetches the physical memory stats for the system in bytes.

    Returns:
      A tuple containing total memory and free memory in the system
      respectively.
    """
    mem_total = None
    mem_free = None
    mem_buffers = None
    mem_cached = None
    mem_total_regex = re.compile('^MemTotal:\\s*(\\d*)\\s*kB')
    mem_free_regex = re.compile('^MemFree:\\s*(\\d*)\\s*kB')
    mem_buffers_regex = re.compile('^Buffers:\\s*(\\d*)\\s*kB')
    mem_cached_regex = re.compile('^Cached:\\s*(\\d*)\\s*kB')
    with files.FileReader('/proc/meminfo') as f:
        for line in f:
            if (m := mem_total_regex.match(line)):
                mem_total = int(m.group(1)) * 1000
            elif (m := mem_free_regex.match(line)):
                mem_free = int(m.group(1)) * 1000
            elif (m := mem_buffers_regex.match(line)):
                mem_buffers = int(m.group(1)) * 1000
            elif (m := mem_cached_regex.match(line)):
                mem_cached = int(m.group(1)) * 1000
    return (mem_total, mem_free + mem_buffers + mem_cached)