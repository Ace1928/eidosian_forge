from .. import errors
from ..constants import IS_WINDOWS_PLATFORM
from ..utils import (
@property
def container_spec(self):
    return self.get('ContainerSpec')