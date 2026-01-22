from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from pluggy import HookspecMarker
@hookspec(historic=True)
def pytest_plugin_registered(plugin: '_PluggyPlugin', plugin_name: str, manager: 'PytestPluginManager') -> None:
    """A new pytest plugin got registered.

    :param plugin: The plugin module or instance.
    :param plugin_name: The name by which the plugin is registered.
    :param manager: The pytest plugin manager.

    .. note::
        This hook is incompatible with hook wrappers.

    Use in conftest plugins
    =======================

    If a conftest plugin implements this hook, it will be called immediately
    when the conftest is registered, once for each plugin registered thus far
    (including itself!), and for all plugins thereafter when they are
    registered.
    """