import os
import time
import mmap
import json
import fnmatch
import asyncio
import itertools
import collections
import logging.handlers
from ray._private.utils import get_or_create_event_loop
from concurrent.futures import ThreadPoolExecutor
from ray._private.utils import run_background_task
from ray.dashboard.modules.event import event_consts
from ray.dashboard.utils import async_loop_forever
def monitor_events(event_dir, callback, monitor_thread_pool_executor: ThreadPoolExecutor, scan_interval_seconds=event_consts.SCAN_EVENT_DIR_INTERVAL_SECONDS, start_mtime=time.time() + event_consts.SCAN_EVENT_START_OFFSET_SECONDS, monitor_files=None, source_types=None):
    """Monitor events in directory. New events will be read and passed to the
    callback.

    Args:
        event_dir: The event log directory.
        callback (def callback(List[str]): pass): A callback accepts a list of
            event strings.
        monitor_thread_pool_executor: A thread pool exector to monitor/update
            events. None means it will use the default execturo which uses
            num_cpus of the machine * 5 threads (before python 3.8) or
            min(32, num_cpus + 5) (from Python 3.8).
        scan_interval_seconds: An interval seconds between two scans.
        start_mtime: Only the event log files whose last modification
            time is greater than start_mtime are monitored.
        monitor_files (Dict[int, MonitorFile]): The map from event log file id
            to MonitorFile object. Monitor all files start from the beginning
            if the value is None.
        source_types (List[str]): A list of source type name from
            event_pb2.Event.SourceType.keys(). Monitor all source types if the
            value is None.
    """
    loop = get_or_create_event_loop()
    if monitor_files is None:
        monitor_files = {}
    logger.info('Monitor events logs modified after %s on %s, the source types are %s.', start_mtime, event_dir, 'all' if source_types is None else source_types)
    MonitorFile = collections.namedtuple('MonitorFile', ['size', 'mtime', 'position'])

    def _source_file_filter(source_file):
        stat = os.stat(source_file)
        return stat.st_mtime > start_mtime

    def _read_monitor_file(file, pos):
        assert isinstance(file, str), f'File should be a str, but a {type(file)}({file}) found'
        fd = os.open(file, os.O_RDONLY)
        try:
            stat = os.stat(fd)
            if stat.st_size <= 0:
                return []
            fid = stat.st_ino or file
            monitor_file = monitor_files.get(fid)
            if monitor_file:
                if monitor_file.position == monitor_file.size and monitor_file.size == stat.st_size and (monitor_file.mtime == stat.st_mtime):
                    logger.debug('Skip reading the file because there is no change: %s', file)
                    return []
                position = monitor_file.position
            else:
                logger.info('Found new event log file: %s', file)
                position = pos
            r = _read_file(fd, position, closefd=False)
            monitor_files[r.fid] = MonitorFile(r.size, r.mtime, r.position)
            loop.call_soon_threadsafe(callback, r.lines)
        except Exception as e:
            raise Exception(f'Read event file failed: {file}') from e
        finally:
            os.close(fd)

    @async_loop_forever(scan_interval_seconds, cancellable=True)
    async def _scan_event_log_files():
        source_files = await loop.run_in_executor(monitor_thread_pool_executor, _get_source_files, event_dir, source_types, _source_file_filter)
        semaphore = asyncio.Semaphore(event_consts.CONCURRENT_READ_LIMIT)

        async def _concurrent_coro(filename):
            async with semaphore:
                return await loop.run_in_executor(monitor_thread_pool_executor, _read_monitor_file, filename, 0)
        await asyncio.gather(*[_concurrent_coro(filename) for filename in list(itertools.chain(*source_files.values()))])
    return run_background_task(_scan_event_log_files())