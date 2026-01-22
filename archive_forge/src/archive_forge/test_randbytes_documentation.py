from __future__ import annotations
from typing import Callable
from typing_extensions import NoReturn, Protocol
from twisted.python import randbytes
from twisted.trial import unittest

        Test C{insecureRandom} without C{random.getrandbits}.
        