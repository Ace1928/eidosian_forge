"""
Stratum prototype simulation engine package.

This package contains the core simulation engine, domain definitions and
utilities for running emergent, layered physical simulations. The goal of
this prototype is to demonstrate a simplified implementation of the
architecture described in the specification provided by the user.

The major subpackages are:

``stratum.core``       Core engine components such as configuration,
                       common types, scheduling, field storage and the
                       event-driven microtick engine (Quanta).
``stratum.domains``    Domain definitions for high‑energy materials,
                       chemistry and other physics submodules.
``stratum.render``     Simple rendering utilities for visualising
                       simulation state.
``stratum.scenarios``  Entry points defining specific simulation
                       scenarios (e.g. uniform stellar gas collapse).

Please see the individual modules for further documentation.
"""

__all__ = [
    "core",
    "domains",
    "render",
    "scenarios",
]