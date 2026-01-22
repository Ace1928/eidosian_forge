import argparse
import asyncio
from datetime import datetime
import importlib
import inspect  # pylint: disable=syntax-error
import io
import json
import collections  # pylint: disable=syntax-error
import os
import signal
import sys
import traceback
import zipfile
from zipimport import zipimporter
import pickle
import uuid
import ansible.module_utils.basic
def create_profiler(self):
    if self.debug_mode:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
        return pr