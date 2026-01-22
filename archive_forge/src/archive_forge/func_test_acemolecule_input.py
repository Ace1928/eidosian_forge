from ase.atoms import Atoms
import numpy as np
from ase.io.acemolecule import read_acemolecule_out, read_acemolecule_input
import pytest
def test_acemolecule_input():
    sample_inputfile = '%% BasicInformation\n    Type Points\n    Scaling 0.35\n    Basis Sinc\n    Grid Basic\n    KineticMatrix Finite_Difference\n    DerivativesOrder 7\n    GeometryFilename acemolecule_test.xyz\n    CellDimensionX 3.37316805\n    CellDimensionY 3.37316805\n    CellDimensionZ 3.37316805\n    PointX 16\n    PointY 16\n    PointZ 16\n    Periodicity 3\n    %% Pseudopotential\n        Pseudopotential 3\n        PSFilePath PATH\n        PSFileSuffix .PBE\n    %% End\n    GeometryFormat xyz\n%% End\n    '
    with open('acemolecule_test.inp', 'w') as fd:
        fd.write(sample_inputfile)
    atoms = Atoms(symbols='HF', positions=np.array([[1.0, 2.0, -0.6], [-1.0, 3.0, 0.7]]))
    atoms.write('acemolecule_test.xyz', format='xyz')
    atoms = read_acemolecule_input('acemolecule_test.inp')
    assert atoms.positions == pytest.approx(np.array([[1.0, 2.0, -0.6], [-1.0, 3.0, 0.7]]))
    assert all(atoms.symbols == 'HF')