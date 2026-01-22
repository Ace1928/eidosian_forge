import dataclasses
import inspect
import enum
import functools
import textwrap
from typing import (
from typing_extensions import Protocol
from cirq import circuits
Show the stored logs >= level in the desired format.

        Args:
            level: The logging level to filter the logs with. The method shows all logs with a
            `LogLevel` >= `level`.
        