from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib
from absl import flags
def recoverable(self):
    """Returns true if the TPU is in a state where training should eventually resume.

    If false the TPU is in a unrecoverable state and should be recreated.
    """
    state = self.state()
    symptoms = self.symptoms()
    if state and state in ['TERMINATED', 'PREEMPTED']:
        return False
    elif FLAGS.runtime_oom_exit and self._oom_event(symptoms):
        return False
    elif FLAGS.hbm_oom_exit and self._hbm_oom_event(symptoms):
        return False
    return True