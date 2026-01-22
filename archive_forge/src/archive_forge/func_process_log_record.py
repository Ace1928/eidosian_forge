import logging
import json
import re
from datetime import date, datetime, time, timezone
import traceback
import importlib
from typing import Any, Dict, Optional, Union, List, Tuple
from inspect import istraceback
from collections import OrderedDict
def process_log_record(self, log_record):
    """
        Override this method to implement custom logic
        on the possibly ordered dictionary.
        """
    return log_record