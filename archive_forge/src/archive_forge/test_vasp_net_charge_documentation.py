import pytest
from ase.build import bulk

    Run VASP tests to ensure that determining number of electrons from
    user-supplied net charge (via the deprecated net_charge parameter) works
    correctly. This is conditional on the existence of the VASP_COMMAND or
    VASP_SCRIPT environment variables.

    This is mainly a slightly reduced duplicate of the vasp_charge test, but with
    flipped signs and with checks that ensure FutureWarning is emitted.

    Should be removed along with the net_charge parameter itself at some point.
    