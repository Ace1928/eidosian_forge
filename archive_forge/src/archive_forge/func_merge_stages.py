from __future__ import annotations
import os
import re
import shutil
import warnings
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import __version__ as CURRENT_VER
from pymatgen.io.core import InputFile
from pymatgen.io.lammps.data import CombinedData, LammpsData
from pymatgen.io.template import TemplateInputGen
def merge_stages(self, stage_names: list[str]) -> None:
    """
        Merges multiple stages of a LammpsInputFile together.
        The merged stage will be at the same index as the first of the stages to be merged.
        The others will appear in the same order as provided in the list. Other non-merged stages will follow.

        Args:
            stage_names (list): list of strings giving the names of the stages to be merged.
        """
    if any((stage not in self.stages_names for stage in stage_names)):
        raise ValueError('At least one of the stages to be merged is not in the LammpsInputFile.')
    indices_stages_to_merge = [self.stages_names.index(stage) for stage in stage_names]
    if not np.all([np.array(indices_stages_to_merge[1:]) >= np.array(indices_stages_to_merge[:-1])]):
        raise ValueError('The provided stages are not in the right order. You should merge stages in the order of appearance\n                in your LammpsInputFile. If you want to reorder stages, modify LammpsInputFile.stages directly. ')
    stages = self.stages[:indices_stages_to_merge[0]]
    merge_name = 'Merge of: ' + ', '.join([self.stages_names[i] for i in indices_stages_to_merge])
    merged_commands = []
    for i in indices_stages_to_merge:
        for j in range(len(self.stages[i]['commands'])):
            merged_commands.append(self.stages[i]['commands'][j])
    merged_stages = {'stage_name': merge_name, 'commands': merged_commands}
    stages.append(merged_stages)
    for i_stage, stage in enumerate(self.stages):
        if i_stage > indices_stages_to_merge[0] and i_stage not in indices_stages_to_merge:
            stages.append(stage)
    self.stages = stages