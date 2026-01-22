from typing import (TYPE_CHECKING, Dict, Iterator, List, Optional, Type, Union,
from . import errors, lock, osutils
from . import revision as _mod_revision
from . import trace
from .inter import InterObject
@property
def supports_file_ids(self):
    """Does this tree support file ids?
        """
    raise NotImplementedError(self.supports_file_ids)