import gc
import os
import warnings
import threading
import contextlib
from abc import ABCMeta, abstractmethod
from ._multiprocessing_helpers import mp
def retrieve_result_callback(self, out):
    try:
        return out.result()
    except ShutdownExecutorError:
        raise RuntimeError("The executor underlying Parallel has been shutdown. This is likely due to the garbage collection of a previous generator from a call to Parallel with return_as='generator'. Make sure the generator is not garbage collected when submitting a new job or that it is first properly exhausted.")