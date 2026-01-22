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
def linefeed(self):
    self.cursor_down()
    self.carriage_return()