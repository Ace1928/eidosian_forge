import importlib
import pkgutil
import re
import sys
import pbr.version
def is_custom(trait):
    """Returns True if the trait string represents a custom trait, or False
    otherwise.

    :param trait: String name of the trait
    """
    return trait.startswith(CUSTOM_NAMESPACE)