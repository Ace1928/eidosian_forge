import logging
import time
import uuid
import os
def get_core_dir():
    import parlai.mturk.core
    return os.path.dirname(os.path.abspath(parlai.mturk.core.dev.__file__))