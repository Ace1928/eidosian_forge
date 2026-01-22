from __future__ import with_statement
import os
import sys
import logging
import threading
import tempfile
import argparse
import Pyro4
from gensim.models import ldamodel
from gensim import utils
Terminate the worker.