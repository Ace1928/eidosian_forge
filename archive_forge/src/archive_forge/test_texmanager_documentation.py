import os
from pathlib import Path
import re
import subprocess
import sys
import matplotlib.pyplot as plt
from matplotlib.texmanager import TexManager
from matplotlib.testing._markers import needs_usetex
import pytest
Test that the preamble is included in the source.