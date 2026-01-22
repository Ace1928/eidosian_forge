import importlib
from ..core.legacy_plugin_wrapper import LegacyPlugin
def partial_legacy_plugin(request):
    return LegacyPlugin(request, legacy_plugin)