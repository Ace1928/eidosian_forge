import logging
import time
import uuid
import os
def get_mturk_dir():
    import parlai.mturk
    return os.path.dirname(os.path.abspath(parlai.mturk.__file__))