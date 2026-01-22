import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase

        Given a top-level package FQPN, L{findMachines} discovers all
        L{MethodicalMachine} instances in and below it.
        