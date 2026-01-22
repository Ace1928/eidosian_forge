import pkgutil
import sys
from _pydev_bundle import pydev_log
def extensions_of_type(extension_type):
    """

    :param T extension_type:  The type of the extension hook
    :rtype: list[T]
    """
    return EXTENSION_MANAGER_INSTANCE.get_extension_classes(extension_type)