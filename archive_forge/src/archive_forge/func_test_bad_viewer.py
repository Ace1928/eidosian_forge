import sys
import pytest
from ase.io import read
from ase.visualize import view
from ase.visualize.external import PyViewer, CLIViewer
from ase.build import bulk
def test_bad_viewer(atoms):
    with pytest.raises(KeyError):
        view(atoms, viewer='_nonexistent_viewer')