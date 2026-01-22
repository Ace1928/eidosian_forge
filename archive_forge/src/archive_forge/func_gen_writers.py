import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import weakref
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.testing.decorators import check_figures_equal
def gen_writers():
    for writer, output in WRITER_OUTPUT:
        if not animation.writers.is_available(writer):
            mark = pytest.mark.skip(f"writer '{writer}' not available on this system")
            yield pytest.param(writer, None, output, marks=[mark])
            yield pytest.param(writer, None, Path(output), marks=[mark])
            continue
        writer_class = animation.writers[writer]
        for frame_format in getattr(writer_class, 'supported_formats', [None]):
            yield (writer, frame_format, output)
            yield (writer, frame_format, Path(output))