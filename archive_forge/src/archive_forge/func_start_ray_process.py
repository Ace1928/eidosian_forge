import base64
import collections
import errno
import io
import json
import logging
import mmap
import multiprocessing
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, IO, AnyStr
import psutil
from filelock import FileLock
import ray
import ray._private.ray_constants as ray_constants
from ray._raylet import GcsClient, GcsClientOptions
from ray.core.generated.common_pb2 import Language
from ray._private.ray_constants import RAY_NODE_IP_FILENAME
def start_ray_process(command: List[str], process_type: str, fate_share: bool, env_updates: Optional[dict]=None, cwd: Optional[str]=None, use_valgrind: bool=False, use_gdb: bool=False, use_valgrind_profiler: bool=False, use_perftools_profiler: bool=False, use_tmux: bool=False, stdout_file: Optional[str]=None, stderr_file: Optional[str]=None, pipe_stdin: bool=False):
    """Start one of the Ray processes.

    TODO(rkn): We need to figure out how these commands interact. For example,
    it may only make sense to start a process in gdb if we also start it in
    tmux. Similarly, certain combinations probably don't make sense, like
    simultaneously running the process in valgrind and the profiler.

    Args:
        command: The command to use to start the Ray process.
        process_type: The type of the process that is being started
            (e.g., "raylet").
        fate_share: If true, the child will be killed if its parent (us) dies.
            True must only be passed after detection of this functionality.
        env_updates: A dictionary of additional environment variables to
            run the command with (in addition to the caller's environment
            variables).
        cwd: The directory to run the process in.
        use_valgrind: True if we should start the process in valgrind.
        use_gdb: True if we should start the process in gdb.
        use_valgrind_profiler: True if we should start the process in
            the valgrind profiler.
        use_perftools_profiler: True if we should profile the process
            using perftools.
        use_tmux: True if we should start the process in tmux.
        stdout_file: A file handle opened for writing to redirect stdout to. If
            no redirection should happen, then this should be None.
        stderr_file: A file handle opened for writing to redirect stderr to. If
            no redirection should happen, then this should be None.
        pipe_stdin: If true, subprocess.PIPE will be passed to the process as
            stdin.

    Returns:
        Information about the process that was started including a handle to
            the process that was started.
    """
    valgrind_env_var = f'RAY_{process_type.upper()}_VALGRIND'
    if os.environ.get(valgrind_env_var) == '1':
        logger.info("Detected environment variable '%s'.", valgrind_env_var)
        use_valgrind = True
    valgrind_profiler_env_var = f'RAY_{process_type.upper()}_VALGRIND_PROFILER'
    if os.environ.get(valgrind_profiler_env_var) == '1':
        logger.info("Detected environment variable '%s'.", valgrind_profiler_env_var)
        use_valgrind_profiler = True
    perftools_profiler_env_var = f'RAY_{process_type.upper()}_PERFTOOLS_PROFILER'
    if os.environ.get(perftools_profiler_env_var) == '1':
        logger.info("Detected environment variable '%s'.", perftools_profiler_env_var)
        use_perftools_profiler = True
    tmux_env_var = f'RAY_{process_type.upper()}_TMUX'
    if os.environ.get(tmux_env_var) == '1':
        logger.info("Detected environment variable '%s'.", tmux_env_var)
        use_tmux = True
    gdb_env_var = f'RAY_{process_type.upper()}_GDB'
    if os.environ.get(gdb_env_var) == '1':
        logger.info("Detected environment variable '%s'.", gdb_env_var)
        use_gdb = True
    if os.environ.get('LD_PRELOAD') is None:
        jemalloc_lib_path = os.environ.get(RAY_JEMALLOC_LIB_PATH, JEMALLOC_SO)
        jemalloc_conf = os.environ.get(RAY_JEMALLOC_CONF)
        jemalloc_comps = os.environ.get(RAY_JEMALLOC_PROFILE)
        jemalloc_comps = [] if not jemalloc_comps else jemalloc_comps.split(',')
        jemalloc_env_vars = propagate_jemalloc_env_var(jemalloc_path=jemalloc_lib_path, jemalloc_conf=jemalloc_conf, jemalloc_comps=jemalloc_comps, process_type=process_type)
    else:
        jemalloc_env_vars = {}
    use_jemalloc_mem_profiler = 'MALLOC_CONF' in jemalloc_env_vars
    if sum([use_gdb, use_valgrind, use_valgrind_profiler, use_perftools_profiler, use_jemalloc_mem_profiler]) > 1:
        raise ValueError("At most one of the 'use_gdb', 'use_valgrind', 'use_valgrind_profiler', 'use_perftools_profiler', and 'use_jemalloc_mem_profiler' flags can be used at a time.")
    if env_updates is None:
        env_updates = {}
    if not isinstance(env_updates, dict):
        raise ValueError("The 'env_updates' argument must be a dictionary.")
    modified_env = os.environ.copy()
    modified_env.update(env_updates)
    if use_gdb:
        if not use_tmux:
            raise ValueError("If 'use_gdb' is true, then 'use_tmux' must be true as well.")
        gdb_init_path = os.path.join(ray._private.utils.get_ray_temp_dir(), f'gdb_init_{process_type}_{time.time()}')
        ray_process_path = command[0]
        ray_process_args = command[1:]
        run_args = ' '.join(["'{}'".format(arg) for arg in ray_process_args])
        with open(gdb_init_path, 'w') as gdb_init_file:
            gdb_init_file.write(f'run {run_args}')
        command = ['gdb', ray_process_path, '-x', gdb_init_path]
    if use_valgrind:
        command = ['valgrind', '--track-origins=yes', '--leak-check=full', '--show-leak-kinds=all', '--leak-check-heuristics=stdstring', '--error-exitcode=1'] + command
    if use_valgrind_profiler:
        command = ['valgrind', '--tool=callgrind'] + command
    if use_perftools_profiler:
        modified_env['LD_PRELOAD'] = os.environ['PERFTOOLS_PATH']
        modified_env['CPUPROFILE'] = os.environ['PERFTOOLS_LOGFILE']
    modified_env.update(jemalloc_env_vars)
    if use_tmux:
        command = ['tmux', 'new-session', '-d', f'{' '.join(command)}']
    if fate_share:
        assert ray._private.utils.detect_fate_sharing_support(), 'kernel-level fate-sharing must only be specified if detect_fate_sharing_support() has returned True'

    def preexec_fn():
        import signal
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        if fate_share and sys.platform.startswith('linux'):
            ray._private.utils.set_kill_on_parent_death_linux()
    win32_fate_sharing = fate_share and sys.platform == 'win32'
    CREATE_SUSPENDED = 4
    if sys.platform == 'win32':
        total_chrs = sum([len(x) for x in command])
        if total_chrs > 31766:
            raise ValueError(f'command is limited to a total of 31767 characters, got {total_chrs}')
    process = ConsolePopen(command, env=modified_env, cwd=cwd, stdout=stdout_file, stderr=stderr_file, stdin=subprocess.PIPE if pipe_stdin else None, preexec_fn=preexec_fn if sys.platform != 'win32' else None, creationflags=CREATE_SUSPENDED if win32_fate_sharing else 0)
    if win32_fate_sharing:
        try:
            ray._private.utils.set_kill_child_on_death_win32(process)
            psutil.Process(process.pid).resume()
        except (psutil.Error, OSError):
            process.kill()
            raise

    def _get_stream_name(stream):
        if stream is not None:
            try:
                return stream.name
            except AttributeError:
                return str(stream)
        return None
    return ProcessInfo(process=process, stdout_file=_get_stream_name(stdout_file), stderr_file=_get_stream_name(stderr_file), use_valgrind=use_valgrind, use_gdb=use_gdb, use_valgrind_profiler=use_valgrind_profiler, use_perftools_profiler=use_perftools_profiler, use_tmux=use_tmux)