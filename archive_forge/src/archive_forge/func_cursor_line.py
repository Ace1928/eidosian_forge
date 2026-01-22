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
def cursor_line(self, line):
    self.cursor.y = min(line, 1) - 1