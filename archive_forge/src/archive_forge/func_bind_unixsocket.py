import os
import socket
from subprocess import Popen, PIPE
from contextlib import contextmanager
import numpy as np
from ase.calculators.calculator import (Calculator, all_changes,
import ase.units as units
from ase.utils import IOContext
from ase.stress import full_3x3_to_voigt_6_stress
@contextmanager
def bind_unixsocket(socketfile):
    assert socketfile.startswith('/tmp/ipi_'), socketfile
    serversocket = socket.socket(socket.AF_UNIX)
    try:
        serversocket.bind(socketfile)
    except OSError as err:
        raise OSError('{}: {}'.format(err, repr(socketfile)))
    try:
        with serversocket:
            yield serversocket
    finally:
        os.unlink(socketfile)