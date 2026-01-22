import re
import time
import numpy as np
from ase.atoms import Atoms
from ase.utils import reader, writer
from ase.cell import Cell

                    NOTE: Have questions about RMC fractionalization matrix for
                          conversion in data2config vs. the ASE matrix.
                          Currently, just support the Cell section.
                    