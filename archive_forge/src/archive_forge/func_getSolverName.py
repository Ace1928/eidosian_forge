import io
import os
import re
import sys
import time
import socket
import base64
import tempfile
import logging
from pyomo.common.dependencies import attempt_import
def getSolverName(self):
    """
        Read in the kestrel_options to pick out the solver name.
        The tricky parts:
          we don't want to be case sensitive, but NEOS is.
          we need to read in options variable
        """
    allKestrelSolvers = self.neos.listSolversInCategory('kestrel')
    kestrelAmplSolvers = []
    for s in allKestrelSolvers:
        i = s.find(':AMPL')
        if i > 0:
            kestrelAmplSolvers.append(s[0:i])
    self.options = None
    if 'kestrel_options' in os.environ:
        self.options = os.getenv('kestrel_options')
    elif 'KESTREL_OPTIONS' in os.environ:
        self.options = os.getenv('KESTREL_OPTIONS')
    if self.options is not None:
        m = re.search('solver\\s*=*\\s*(\\S+)', self.options, re.IGNORECASE)
        NEOS_solver_name = None
        if m:
            solver_name = m.groups()[0]
            for s in kestrelAmplSolvers:
                if s.upper() == solver_name.upper():
                    NEOS_solver_name = s
                    break
            if not NEOS_solver_name:
                raise RuntimeError('%s is not available on NEOS.  Choose from:\n\t%s' % (solver_name, '\n\t'.join(kestrelAmplSolvers)))
    if self.options is None or m is None:
        raise RuntimeError('%s is not available on NEOS.  Choose from:\n\t%s' % (solver_name, '\n\t'.join(kestrelAmplSolvers)))
    return NEOS_solver_name