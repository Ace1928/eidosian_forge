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
@pytest.mark.parametrize('writer, frame_format, output', gen_writers())
def test_grabframe(tmpdir, writer, frame_format, output):
    WriterClass = animation.writers[writer]
    if frame_format is not None:
        plt.rcParams['animation.frame_format'] = frame_format
    fig, ax = plt.subplots()
    dpi = None
    codec = None
    if writer == 'ffmpeg':
        fig.set_size_inches((10.85, 9.21))
        dpi = 100.0
        codec = 'h264'
    test_writer = WriterClass()
    with tmpdir.as_cwd():
        with test_writer.saving(fig, output, dpi):
            test_writer.grab_frame()
            for k in {'dpi', 'bbox_inches', 'format'}:
                with pytest.raises(TypeError, match=f'grab_frame got an unexpected keyword argument {k!r}'):
                    test_writer.grab_frame(**{k: object()})