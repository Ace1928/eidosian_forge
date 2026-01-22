import os
import time
import subprocess
import re
import warnings
import numpy as np
from ase.geometry import cell_to_cellpar
from ase.calculators.calculator import (FileIOCalculator, Calculator, equal,
from ase.calculators.openmx.parameters import OpenMXParameters
from ase.calculators.openmx.default_settings import default_dictionary
from ase.calculators.openmx.reader import read_openmx, get_file_name
from ase.calculators.openmx.writer import write_openmx
def initialize_output_setting(self, **kwargs):
    output_setting = {}
    self.output_setting = dict(self.default_output_setting)
    for key, value in kwargs.items():
        if key in self.default_output_setting:
            output_setting[key] = value
    self.output_setting.update(output_setting)
    self.__dict__.update(self.output_setting)