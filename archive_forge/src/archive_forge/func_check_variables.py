import copy
import glob
import inspect
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime
from numbers import Number
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
import numpy as np
import psutil
import ray
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.air._internal.json import SafeFallbackEncoder  # noqa
from ray.air._internal.util import (  # noqa: F401
from ray._private.dict import (  # noqa: F401
def check_variables(objects, failure_set, printer):
    for var_name, variable in objects.items():
        msg = None
        try:
            _check_serializability(var_name, variable)
            status = 'PASSED'
        except Exception as e:
            status = 'FAILED'
            msg = f'{e.__class__.__name__}: {str(e)}'
            failure_set.add(var_name)
        printer(f"{str(variable)}[name='{var_name}'']... {status}")
        if msg:
            printer(msg)