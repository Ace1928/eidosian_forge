from copy import deepcopy
import csv, math, os
from nibabel import load
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list
from ..utils.misc import normalize_mc_params
from .. import config, logging
def gen_info(run_event_files):
    """Generate subject_info structure from a list of event files."""
    info = []
    for i, event_files in enumerate(run_event_files):
        runinfo = Bunch(conditions=[], onsets=[], durations=[], amplitudes=[])
        for event_file in event_files:
            _, name = os.path.split(event_file)
            if '.run' in name:
                name, _ = name.split('.run%03d' % (i + 1))
            elif '.txt' in name:
                name, _ = name.split('.txt')
            runinfo.conditions.append(name)
            event_info = np.atleast_2d(np.loadtxt(event_file))
            runinfo.onsets.append(event_info[:, 0].tolist())
            if event_info.shape[1] > 1:
                runinfo.durations.append(event_info[:, 1].tolist())
            else:
                runinfo.durations.append([0])
            if event_info.shape[1] > 2:
                runinfo.amplitudes.append(event_info[:, 2].tolist())
            else:
                delattr(runinfo, 'amplitudes')
        info.append(runinfo)
    return info