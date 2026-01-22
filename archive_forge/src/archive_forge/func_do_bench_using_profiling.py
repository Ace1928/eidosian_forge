from __future__ import annotations
import collections
import contextlib
import enum
import functools
import getpass
import inspect
import itertools
import logging
import math
import operator
import os
import platform
import re
import shutil
import sys
import tempfile
import textwrap
import time
import unittest
from io import StringIO
from typing import (
from unittest import mock
import sympy
from typing_extensions import Concatenate, ParamSpec
import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch.autograd import DeviceType
from torch.autograd.profiler_util import EventList
from torch.utils._sympy.functions import CeilDiv, CleanDiv, FloorDiv, ModularIndexing
from . import config
def do_bench_using_profiling(fn: Callable[[], Any], warmup=25, rep=100) -> float:
    """
    Returns benchmark results by examining torch profiler events.
    This could be more accurate as it doesn't count CPU side overhead.
    However, this also requires manually excluding irrelevant event, e.g.
    vectorized_elementwise_kernel which is used to fill L2 cache,
    various CUDA events, etc, so could also be fragile.
    """
    fn()
    torch.cuda.synchronize()
    cache = torch.empty(int(256000000.0 // 4), dtype=torch.int, device='cuda')
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    for _ in range(n_warmup):
        fn()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        for i in range(n_repeat):
            cache.zero_()
            fn()
        torch.cuda.synchronize()
    log.debug('raw events')
    log.debug(p.key_averages().table(sort_by='self_cuda_time_total', row_limit=-1))
    filtered_events = EventList([event for event in p.events() if event.device_type == DeviceType.CUDA and event.name != 'Context Sync'])
    if len(filtered_events) % n_repeat != 0:
        raise RuntimeError('Failed to divide all profiling events into #repeat groups. #CUDA events: %d, #repeats: %s', len(filtered_events), n_repeat)
    num_event_per_group = len(filtered_events) / n_repeat
    actual_events = EventList([event for i, event in enumerate(filtered_events) if i % num_event_per_group != 0])
    actual_events._build_tree()
    actual_events = actual_events.key_averages()
    log.debug('profiling time breakdown')
    log.debug(actual_events.table(row_limit=-1))
    res = sum((event.cuda_time_total for event in actual_events)) / 1000.0 / n_repeat
    log.debug('profiling results: %s ms', res)
    return res