from debtcollector import removals
import sqlalchemy
from stevedore import enabled
from oslo_db import exception
def check_plugin_enabled(ext):
    """Used for EnabledExtensionManager."""
    return ext.obj.enabled