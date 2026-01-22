from __future__ import absolute_import
import sys
import subprocess
import time
from twisted.trial.unittest import TestCase
from crochet._shutdown import (
from ..tests import crochet_directory
import threading, sys
from crochet._shutdown import register, _watchdog

        Registered functions that raise an error have the error logged, and
        run() continues processing.
        