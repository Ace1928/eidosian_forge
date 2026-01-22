from ase import io as aseIO
from ase.io.dlp4 import iread_dlp_history
from io import StringIO
import numpy as np
def test_dlp2():
    mol = aseIO.read(fd2, format='dlp4', symbols=['C', 'Cl', 'H', 'H', 'H'])
    assert (mol.get_array('dlp4_labels') == np.array(['1', '', '', '', '1'])).all()