"""
Scenarios package for the Stratum simulation.

This package contains complete simulation scenarios that demonstrate
the Stratum engine's capabilities:

- ``stellar_collapse``: Basic stellar gas collapse simulation
- ``stellar_collapse_runtime``: Wall-clock duration controlled simulation
- ``stellar_screensaver``: Real-time Pygame visualization

Each scenario module can be run directly as a script or imported
to use programmatically.

Example usage::

    # Run stellar collapse scenario
    python -m scenarios.stellar_collapse

    # Import and customize
    from scenarios.stellar_collapse import run_stellar_collapse
    run_stellar_collapse(grid_size=64, num_ticks=1000)
"""

__all__ = [
    "stellar_collapse",
    "stellar_collapse_runtime",
    "stellar_screensaver",
]
