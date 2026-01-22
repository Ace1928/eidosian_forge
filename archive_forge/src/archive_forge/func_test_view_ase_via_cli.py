import sys
import pytest
from ase.io import read
from ase.visualize import view
from ase.visualize.external import PyViewer, CLIViewer
from ase.build import bulk
def test_view_ase_via_cli(atoms):
    viewer = view(atoms, viewer='ase_gui_cli')
    assert viewer.poll() is None
    viewer.terminate()
    status = viewer.wait()
    assert status != 0