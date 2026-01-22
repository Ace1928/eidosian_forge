from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import unittest
from gc import collect
from gc import get_objects
from threading import active_count as active_thread_count
from time import sleep
from time import time
import psutil
from greenlet import greenlet as RawGreenlet
from greenlet import getcurrent
from greenlet._greenlet import get_pending_cleanup_count
from greenlet._greenlet import get_total_main_greenlets
from . import leakcheck
def wait_for_pending_cleanups(self, initial_active_threads=None, initial_main_greenlets=None):
    initial_active_threads = initial_active_threads or self.threads_before_test
    initial_main_greenlets = initial_main_greenlets or self.main_greenlets_before_test
    sleep_time = self.cleanup_attempt_sleep_duration
    sleep(sleep_time)
    quit_after = time() + self.cleanup_max_sleep_seconds
    while get_pending_cleanup_count() or active_thread_count() > initial_active_threads or (not self.expect_greenlet_leak and get_total_main_greenlets() > initial_main_greenlets):
        sleep(sleep_time)
        if time() > quit_after:
            print('Time limit exceeded.')
            print('Threads: Waiting for only', initial_active_threads, '-->', active_thread_count())
            print('MGlets : Waiting for only', initial_main_greenlets, '-->', get_total_main_greenlets())
            break
    collect()