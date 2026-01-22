import logging
import json
import re
from datetime import date, datetime, time, timezone
import traceback
import importlib
from typing import Any, Dict, Optional, Union, List, Tuple
from inspect import istraceback
from collections import OrderedDict
def format_datetime_obj(self, obj):
    return obj.isoformat()