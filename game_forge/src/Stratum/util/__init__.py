"""
Utility subpackage for Stratum.

This package exposes miscellaneous helpers that are useful across the
Stratum codebase.  The most notable utility currently provided is
``ClassRegistry``, a tool that allows introspection of modules,
classes and functions defined within the ``stratum`` package.  See
``stratum.util.classregistry`` for details.
"""

from .classregistry import ClassRegistry

__all__ = ['ClassRegistry']