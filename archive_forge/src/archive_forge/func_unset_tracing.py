from argparse import Action
import os
import sys
from typing import Generator
from typing import List
from typing import Optional
from typing import Union
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import PrintHelp
from _pytest.config.argparsing import Parser
from _pytest.terminal import TerminalReporter
import pytest
def unset_tracing() -> None:
    debugfile.close()
    sys.stderr.write('wrote pytest debug information to %s\n' % debugfile.name)
    config.trace.root.setwriter(None)
    undo_tracing()