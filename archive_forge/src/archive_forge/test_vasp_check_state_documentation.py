import os
import pytest

    Run tests to ensure that the VASP check_state() function call works correctly,
    i.e. correctly sets the working directories and works in that directory.

    This is conditional on the existence of the VASP_COMMAND or VASP_SCRIPT
    environment variables

    