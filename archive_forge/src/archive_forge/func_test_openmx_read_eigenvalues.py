import io
import numpy as np
from ase import Atoms
from ase.units import Ha, Bohr
from ase.calculators.openmx.reader import read_openmx, read_eigenvalues
def test_openmx_read_eigenvalues():
    tol = 0.01
    eigenvalues_pattern = 'Eigenvalues (Hartree)'
    with io.StringIO(openmx_eigenvalues_gamma_sample) as fd:
        while True:
            line = fd.readline()
            if eigenvalues_pattern in line:
                break
        eigenvalues = read_eigenvalues(line, fd)
    gamma_eigenvalues = np.array([[[-0.96233478518931, -0.96233478518931], [-0.9418933985645, -0.9418933985645], [-0.86350555424836, -0.86350555424836], [-0.83918201748919, -0.83918201748919], [-0.72288697309928, -0.72288697309928], [-0.67210805969879, -0.67210805969879], [-0.64903406048465, -0.64903406048465], [-0.58249976216367, -0.58249976216367], [-0.55161386332358, -0.55161386332358]]])
    gamma_eigenvalues = np.swapaxes(gamma_eigenvalues.T, 1, 2)
    assert np.all(np.isclose(eigenvalues, gamma_eigenvalues, atol=tol))
    with io.StringIO(openmx_eigenvalues_bulk_sample) as fd:
        while True:
            line = fd.readline()
            if eigenvalues_pattern in line:
                break
        eigenvalues = read_eigenvalues(line, fd)
    bulk_eigenvalues = np.array([[[-2.33424746491277, -2.3342474691788], [-2.33424055817432, -2.33424056243807], [-2.33419668076232, -2.33419668261398], [-1.46440634271635, -1.46440634435648], [-1.46439286017722, -1.46439286180118], [-1.46436086583111, -1.46436086399066], [-1.46397017044962, -1.46397017874325], [-1.46394407220255, -1.46394408049882], [-1.46389030794971, -1.46389031384386]], [[-2.3342470525902, -2.33424705685571], [-2.33424133604313, -2.33424134030309], [-2.33419651862703, -2.33419652048304], [-1.46440529840756, -1.46440530004421], [-1.46439446518585, -1.46439446677862], [-1.46436032862668, -1.46436032682027], [-1.46396740984959, -1.46396741813205], [-1.463946382109, -1.46394639039694], [-1.46389029838585, -1.46389030429995]]])
    bulk_eigenvalues = np.swapaxes(bulk_eigenvalues.T, 1, 2)
    assert np.all(np.isclose(eigenvalues[:, :2, :], bulk_eigenvalues, atol=tol))