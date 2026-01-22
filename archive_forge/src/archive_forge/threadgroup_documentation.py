import logging
import threading
import warnings
from debtcollector import removals
import eventlet
from eventlet import greenpool
from oslo_service import loopingcall
from oslo_utils import timeutils
Cancel unstarted threads in the group, and optionally stop the rest.

        .. warning::
            This method is deprecated and should not be used. It will be
            removed in a future release.

        If called without the ``timeout`` argument, this does **not** stop any
        running threads, but prevents any threads in the group that have not
        yet started from running, then returns immediately. Timers are not
        affected.

        If the 'timeout' argument is supplied, then it serves as a grace period
        to allow running threads to finish. After the timeout, any threads in
        the group that are still running will be killed by raising GreenletExit
        in them, and all timers will be stopped (so that they are not
        retriggered - timer calls that are in progress will not be
        interrupted). This method will **not** block until all threads have
        actually exited, nor that all in-progress timer calls have completed.
        To guarantee that all threads have exited, call :func:`wait`. If all
        threads complete before the timeout expires, timers will be left
        running; there is no way to then stop those timers, so for consistent
        behaviour :func`stop_timers` should be called before calling this
        method.

        :param throw_args: the `exc_info` data to raise from
                           :func:`Thread.wait` for any of the unstarted
                           threads. (Though note that :func:`ThreadGroup.wait`
                           suppresses exceptions.)
        :param timeout: time to wait for running threads to complete before
                        calling stop(). If not supplied, threads that are
                        already running continue to completion.
        :param wait_time: length of time in seconds to sleep between checks of
                          whether any threads are still alive. (Default 1s.)
        