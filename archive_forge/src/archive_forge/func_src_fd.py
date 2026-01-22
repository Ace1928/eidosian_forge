import itertools
import logging
import os
import queue
import re
import signal
import struct
import sys
import threading
import time
from collections import defaultdict
import wandb
@property
def src_fd(self):
    return self.src_stream.fileno()