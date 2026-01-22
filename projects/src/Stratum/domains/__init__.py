"""
Domains package for Stratum engine.

This package contains domain specific logic layered on top of the
core engine. Each subpackage under ``domains`` implements a layer
such as materials (high energy physics) or chemistry (low energy
interactions). Domain modules define their own classes and helper
functions that operate on the state stored in the ``Fabric``. The
engine orchestrates these modules via the ``Quanta`` subsystem.
"""

__all__ = ["materials", "chemistry"]