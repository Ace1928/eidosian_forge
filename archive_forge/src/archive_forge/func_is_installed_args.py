from .config import load_config_schema
from .utils import ShellSpec
@property
def is_installed_args(self):
    return ['-e', f"cat(system.file(package='{self.package}'))"]