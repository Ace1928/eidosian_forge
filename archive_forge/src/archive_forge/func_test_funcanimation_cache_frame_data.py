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
@pytest.mark.parametrize('cache_frame_data', [False, True])
def test_funcanimation_cache_frame_data(cache_frame_data):
    fig, ax = plt.subplots()
    line, = ax.plot([], [])

    class Frame(dict):
        pass

    def init():
        line.set_data([], [])
        return (line,)

    def animate(frame):
        line.set_data(frame['x'], frame['y'])
        return (line,)
    frames_generated = []

    def frames_generator():
        for _ in range(5):
            x = np.linspace(0, 10, 100)
            y = np.random.rand(100)
            frame = Frame(x=x, y=y)
            frames_generated.append(weakref.ref(frame))
            yield frame
    MAX_FRAMES = 100
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames_generator, cache_frame_data=cache_frame_data, save_count=MAX_FRAMES)
    writer = NullMovieWriter()
    anim.save('unused.null', writer=writer)
    assert len(frames_generated) == 5
    np.testing.break_cycles()
    for f in frames_generated:
        assert (f() is None) != cache_frame_data