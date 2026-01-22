from collections.abc import Mapping, Sequence
from subprocess import check_call, DEVNULL
from os import unlink
from pathlib import Path
import numpy as np
from ase.io.utils import PlottingVariables
from ase.constraints import FixAtoms
from ase import Atoms
def write_ini(self, path):
    """Write ini file."""
    ini_str = f'Input_File_Name={path.with_suffix('.pov').name}\nOutput_to_File=True\nOutput_File_Type=N\nOutput_Alpha={('on' if self.transparent else 'off')}\n; if you adjust Height, and width, you must preserve the ratio\n; Width / Height = {self.canvas_width / self.canvas_height:f}\nWidth={self.canvas_width}\nHeight={self.canvas_height}\nAntialias=True\nAntialias_Threshold=0.1\nDisplay={self.display}\nPause_When_Done={self.pause}\nVerbose=False\n'
    with open(path, 'w') as fd:
        fd.write(ini_str)
    return path