import numpy as np
from ase import io
from ase import build
from ase.io.espresso import parse_position_line
from pytest import approx
def test_pw_output():
    """Read pw output file."""
    with open('pw_output.pwo', 'w') as pw_output_f:
        pw_output_f.write(pw_output_text)
    pw_output_traj = io.read('pw_output.pwo', index=':')
    assert len(pw_output_traj) == 2
    assert pw_output_traj[1].get_volume() > pw_output_traj[0].get_volume()