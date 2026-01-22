import configobj
from . import bedding, cmdline, errors, globbing, osutils
def rules_path():
    """Return the default rules file path."""
    return osutils.pathjoin(bedding.config_dir(), 'rules')