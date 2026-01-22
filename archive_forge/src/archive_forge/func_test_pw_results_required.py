import numpy as np
from ase import io
from ase import build
from ase.io.espresso import parse_position_line
from pytest import approx
def test_pw_results_required():
    """Check only configurations with results are read unless requested."""
    with open('pw_output.pwo', 'w') as pw_output_f:
        pw_output_f.write(pw_output_text)
    pw_output_traj = io.read('pw_output.pwo', index=':')
    assert 'energy' in pw_output_traj[-1].calc.results
    assert len(pw_output_traj) == 2
    pw_output_traj = io.read('pw_output.pwo', index=':', results_required=False)
    assert len(pw_output_traj) == 3
    assert 'energy' not in pw_output_traj[-1].calc.results
    pw_output_config = io.read('pw_output.pwo')
    assert 'energy' in pw_output_config.calc.results
    pw_output_config = io.read('pw_output.pwo', results_required=False)
    assert 'energy' not in pw_output_config.calc.results