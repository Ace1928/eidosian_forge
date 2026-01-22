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
@pytest.mark.parametrize('anim', [dict(klass=dict)], indirect=['anim'])
def test_save_count_override_warnings_has_length(anim):
    save_count = 5
    frames = list(range(2))
    match_target = f'You passed in an explicit save_count={save_count!r} which is being ignored in favor of len(frames)={len(frames)!r}.'
    with pytest.warns(UserWarning, match=re.escape(match_target)):
        anim = animation.FuncAnimation(**{**anim, 'frames': frames, 'save_count': save_count})
    assert anim._save_count == len(frames)
    anim._init_draw()