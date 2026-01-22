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
def insert_lines(self, n=1):
    for i in range(self.num_lines - 1, self.cursor.y, -1):
        self.buffer[i + n] = self.buffer[i]
    for i in range(self.cursor.y + 1, self.cursor.y + 1 + n):
        if i in self.buffer:
            del self.buffer[i]