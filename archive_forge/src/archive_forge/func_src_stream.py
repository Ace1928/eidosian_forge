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
def src_stream(self):
    return getattr(sys, '__%s__' % self.src)