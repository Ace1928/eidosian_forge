import os
import sys
import logging
import argparse
import threading
import tempfile
import queue as Queue
import Pyro4
from gensim.models import lsimodel
from gensim import utils
Terminate the worker.