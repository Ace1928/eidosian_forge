from collections import deque, OrderedDict
from typing import Union, Optional, Set, Any, Dict, List, Tuple
from datetime import timedelta
import functools
import math
import time
import re
import shutil
import json
from parlai.core.message import Message
from parlai.utils.strings import colorize
import parlai.utils.logging as logging
def clip_text(text, max_len):
    """
    Clip text to max length, adding ellipses.
    """
    if len(text) > max_len:
        begin_text = ' '.join(text[:math.floor(0.8 * max_len)].split(' ')[:-1])
        end_text = ' '.join(text[len(text) - math.floor(0.2 * max_len):].split(' ')[1:])
        if len(end_text) > 0:
            text = begin_text + ' ...\n' + end_text
        else:
            text = begin_text + ' ...'
    return text