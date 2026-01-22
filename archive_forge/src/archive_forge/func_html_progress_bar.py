import re
import time
from typing import Optional
import IPython.display as disp
from ..trainer_callback import TrainerCallback
from ..trainer_utils import IntervalStrategy, has_length
def html_progress_bar(value, total, prefix, label, width=300):
    return f"\n    <div>\n      {prefix}\n      <progress value='{value}' max='{total}' style='width:{width}px; height:20px; vertical-align: middle;'></progress>\n      {label}\n    </div>\n    "