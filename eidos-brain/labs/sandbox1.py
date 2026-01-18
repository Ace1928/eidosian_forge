#!/usr/bin/env python
"""Sandbox for quick experiments."""

from core.eidos_core import EidosCore

if __name__ == "__main__":
    core = EidosCore()
    core.process_cycle("initial experience")
    print(core.reflect())
