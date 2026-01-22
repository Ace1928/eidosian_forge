import os
import numpy as np
from ase.io.octopus.input import (
from ase.io.octopus.output import read_eigenvalues_file, read_static_info
from ase.calculators.calculator import (
Read octopus output files and extract data.