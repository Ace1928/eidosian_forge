import logging
import queue
import sys
import threading
import time
from typing import TYPE_CHECKING, Optional, Tuple, Type, Union
from ..lib import tracelog
Class to manage reading from queues safely.