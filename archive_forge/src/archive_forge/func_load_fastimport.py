from ... import version_info  # noqa: F401
from ...commands import plugin_cmds
def load_fastimport():
    """Load the fastimport module or raise an appropriate exception."""
    try:
        import fastimport
    except ModuleNotFoundError as e:
        from ...errors import DependencyNotPresent
        raise DependencyNotPresent('fastimport', 'fastimport requires the fastimport python module')
    if fastimport.__version__ < (0, 9, 8):
        from ...errors import DependencyNotPresent
        raise DependencyNotPresent('fastimport', 'fastimport requires at least version 0.9.8 of the fastimport python module')