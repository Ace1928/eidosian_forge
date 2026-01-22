import subprocess
import os
import sys
import random
import threading
import collections
import logging
import time
def is_in_databricks_runtime():
    return 'DATABRICKS_RUNTIME_VERSION' in os.environ