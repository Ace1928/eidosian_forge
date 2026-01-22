import abc
from neutron_lib.api.definitions import portbindings
def supported_extensions(self, extensions):
    """Return the mechanism driver supported extensions

        By default this method will return the same provided set, without any
        filtering. In case any particular mechanism driver needs to filter out
        any specific extension or supports only a reduced set of extensions,
        this method should be override.

        :param extensions: set of extensions supported by the instance that
                           created this mechanism driver.
        :returns: a set of the extensions currently supported by this
                  mechanism driver
        """
    return extensions