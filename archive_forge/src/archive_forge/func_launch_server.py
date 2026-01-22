import os
import socket
from subprocess import Popen, PIPE
from contextlib import contextmanager
import numpy as np
from ase.calculators.calculator import (Calculator, all_changes,
import ase.units as units
from ase.utils import IOContext
from ase.stress import full_3x3_to_voigt_6_stress
def launch_server(self):
    return self.closelater(SocketServer(port=self._port, unixsocket=self._unixsocket, timeout=self.timeout, log=self.log))