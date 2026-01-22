import codecs
import os
import threading
from debugpy import launcher
from debugpy.common import log
def wait_for_remaining_output():
    """Waits for all remaining output to be captured and propagated."""
    for category, instance in CaptureOutput.instances.items():
        log.info('Waiting for remaining {0} of {1}.', category, instance._whose)
        instance._worker_thread.join()