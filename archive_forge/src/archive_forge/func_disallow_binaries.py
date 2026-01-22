from typing import FrozenSet, Optional, Set
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.exceptions import CommandError
def disallow_binaries(self) -> None:
    self.handle_mutual_excludes(':all:', self.no_binary, self.only_binary)