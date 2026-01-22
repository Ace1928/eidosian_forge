import os
import socket
from subprocess import Popen, PIPE
from contextlib import contextmanager
import numpy as np
from ase.calculators.calculator import (Calculator, all_changes,
import ase.units as units
from ase.utils import IOContext
from ase.stress import full_3x3_to_voigt_6_stress
Initialize socket I/O calculator.

        This calculator launches a server which passes atomic
        coordinates and unit cells to an external code via a socket,
        and receives energy, forces, and stress in return.

        ASE integrates this with the Quantum Espresso, FHI-aims and
        Siesta calculators.  This works with any external code that
        supports running as a client over the i-PI protocol.

        Parameters:

        calc: calculator or None

            If calc is not None, a client process will be launched
            using calc.command, and the input file will be generated
            using ``calc.write_input()``.  Otherwise only the server will
            run, and it is up to the user to launch a compliant client
            process.

        port: integer

            port number for socket.  Should normally be between 1025
            and 65535.  Typical ports for are 31415 (default) or 3141.

        unixsocket: str or None

            if not None, ignore host and port, creating instead a
            unix socket using this name prefixed with ``/tmp/ipi_``.
            The socket is deleted when the calculator is closed.

        timeout: float >= 0 or None

            timeout for connection, by default infinite.  See
            documentation of Python sockets.  For longer jobs it is
            recommended to set a timeout in case of undetected
            client-side failure.

        log: file object or None (default)

            logfile for communication over socket.  For debugging or
            the curious.

        In order to correctly close the sockets, it is
        recommended to use this class within a with-block:

        >>> with SocketIOCalculator(...) as calc:
        ...    atoms.calc = calc
        ...    atoms.get_forces()
        ...    atoms.rattle()
        ...    atoms.get_forces()

        It is also possible to call calc.close() after
        use.  This is best done in a finally-block.