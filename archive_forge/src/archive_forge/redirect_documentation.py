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
# Arguments.

        `src`: Source stream to be redirected. "stdout" or "stderr".
        `cbs`: tuple/list of callbacks. Each callback should take exactly 1 argument (bytes).

        