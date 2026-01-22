import pytest
from ase.md.contour_exploration import ContourExploration
import numpy as np
from ase import io
from .test_ce_potentiostat import Al_block, bulk_Al_settings
This test ensures that logging to a text file and to the trajectory file are
reporting the same values as in the ContourExploration object.