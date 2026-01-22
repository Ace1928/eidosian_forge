import datetime
import errno
import html
import json
import os
import random
import shlex
import textwrap
import time
from tensorboard import manager
def print_or_update(message):
    if handle is None:
        print(message)
    else:
        handle.update(IPython.display.Pretty(message))