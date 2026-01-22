import abc
from neutron_lib.api.definitions import portbindings
@property
def extension_aliases(self):
    """List of extension aliases supported by the driver.

        Return a list of aliases identifying the core API extensions
        supported by the driver. By default this just returns the
        extension_alias property for backwards compatibility.
        """
    return [self.extension_alias]