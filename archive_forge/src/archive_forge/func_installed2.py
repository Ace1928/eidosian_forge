import pytest
import os
def installed2():
    for env in ['VASP_COMMAND', 'VASP_SCRIPT', 'ASE_VASP_COMMAND']:
        if os.getenv(env):
            break
    else:
        pytest.skip('Neither ASE_VASP_COMMAND, VASP_COMMAND nor VASP_SCRIPT defined')
    return True