from ase import io as aseIO
from ase.io.dlp4 import iread_dlp_history
from io import StringIO
import numpy as np
def test_dlp3():
    cells = []
    cells.append(np.array([[23.9999973028, 0.0, 0.0], [0.0, 23.9999973028, 0.0], [0.0, 0.0, 23.9999973028]]))
    cells.append(np.array([[23.9999947494, 0.0, 0.0], [0.0, 23.9999947494, 0.0], [0.0, 0.0, 23.9999947494]]))
    cells.append(np.array([[23.9999911871, 0.0, 0.0], [0.0, 23.9999911871, 0.0], [0.0, 0.0, 23.9999911871]]))
    traj = aseIO.read(fd3, format='dlp-history', index=slice(0, None))
    assert len(traj) == 3
    traj = aseIO.iread(fd3, format='dlp-history', index=slice(0, None))
    for i, frame in enumerate(traj):
        assert len(frame) == 4
        assert all(frame.symbols == 'OHHX')
        assert np.isclose(frame.get_cell(), cells[i]).all()
    symbols = frame.get_chemical_symbols()
    traj = iread_dlp_history(fd3, symbols)
    for i, frame in enumerate(traj):
        assert len(frame) == 4
        assert all(frame.symbols == 'OHHX')
        assert np.isclose(frame.get_cell(), cells[i]).all()
        assert frame.has('initial_charges')