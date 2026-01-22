import os
import math
import random
from pathlib import Path
import pytest
from IPython.utils import text
def test_eval_formatter():
    f = text.EvalFormatter()
    eval_formatter_check(f)
    eval_formatter_no_slicing_check(f)