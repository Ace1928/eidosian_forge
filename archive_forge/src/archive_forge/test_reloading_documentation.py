from numpy.testing import (
from numpy.compat import pickle
import pytest
import sys
import subprocess
import textwrap
from importlib import reload
At the time of writing this, it is *not* truly supported, but
    apparently enough users rely on it, for it to be an annoying change
    when it started failing previously.
    