from ... import version_info  # noqa: F401
from ... import commands, config, hooks
def register_option(key, member):
    """Lazily register an option."""
    config.option_registry.register_lazy(key, 'breezy.plugins.upload.cmds', member)