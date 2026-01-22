import abc
from typing import List
import stevedore
def unitary_synthesis_plugin_names():
    """Return a list of installed unitary synthesis plugin names

    Returns:
        list: A list of the installed unitary synthesis plugin names. The plugin names are valid
        values for the :func:`~qiskit.compiler.transpile` kwarg ``unitary_synthesis_method``.
    """
    plugins = UnitarySynthesisPluginManager()
    return plugins.ext_plugins.names()