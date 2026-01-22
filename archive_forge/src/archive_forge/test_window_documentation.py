from __future__ import annotations
from typing import Callable
from twisted.conch.insults.window import ScrolledArea, TextOutput, TopWindow
from twisted.trial.unittest import TestCase

        The parent of the widget passed to L{ScrolledArea} is set to a new
        L{Viewport} created by the L{ScrolledArea} which itself has the
        L{ScrolledArea} instance as its parent.
        