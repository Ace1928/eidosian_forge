from __future__ import annotations
import re
from glob import glob
from io import StringIO
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable
from pymatgen.io.lammps.data import LammpsBox
def parse_lammps_log(filename: str='log.lammps') -> list[pd.DataFrame]:
    """
    Parses log file with focus on thermo data. Both one and multi line
    formats are supported. Any incomplete runs (no "Loop time" marker)
    will not be parsed.

    Notes:
        SHAKE stats printed with thermo data are not supported yet.
        They are ignored in multi line format, while they may cause
        issues with dataframe parsing in one line format.

    Args:
        filename (str): Filename to parse.

    Returns:
        [pd.DataFrame] containing thermo data for each completed run.
    """
    with zopen(filename, mode='rt') as file:
        lines = file.readlines()
    begin_flag = ('Memory usage per processor =', 'Per MPI rank memory allocation (min/avg/max) =')
    end_flag = 'Loop time of'
    begins, ends = ([], [])
    for idx, line in enumerate(lines):
        if line.startswith(begin_flag):
            begins.append(idx)
        elif line.startswith(end_flag):
            ends.append(idx)

    def _parse_thermo(lines: list[str]) -> pd.DataFrame:
        multi_pattern = '-+\\s+Step\\s+([0-9]+)\\s+-+'
        if re.match(multi_pattern, lines[0]):
            timestep_marks = [idx for idx, line in enumerate(lines) if re.match(multi_pattern, line)]
            timesteps = np.split(lines, timestep_marks)[1:]
            dicts = []
            kv_pattern = '([0-9A-Za-z_\\[\\]]+)\\s+=\\s+([0-9eE\\.+-]+)'
            for ts in timesteps:
                data = {}
                step = re.match(multi_pattern, ts[0])
                assert step is not None
                data['Step'] = int(step[1])
                data.update({k: float(v) for k, v in re.findall(kv_pattern, ''.join(ts[1:]))})
                dicts.append(data)
            df = pd.DataFrame(dicts)
            columns = ['Step'] + [k for k, v in re.findall(kv_pattern, ''.join(timesteps[0][1:]))]
            df = df[columns]
        else:
            df = pd.read_csv(StringIO(''.join(lines)), delim_whitespace=True)
        return df
    runs = []
    for b, e in zip(begins, ends):
        runs.append(_parse_thermo(lines[b + 1:e]))
    return runs