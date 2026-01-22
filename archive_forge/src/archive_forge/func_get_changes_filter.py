from __future__ import annotations
import typing as t
from .io import (
from .util import (
from .ci import (
from .classification import (
from .config import (
from .metadata import (
from .provisioning import (
def get_changes_filter(args: TestConfig) -> list[str]:
    """Return a list of targets which should be tested based on the changes made."""
    paths = detect_changes(args)
    if not args.metadata.change_description:
        if paths:
            changes = categorize_changes(args, paths, args.command)
        else:
            changes = ChangeDescription()
        args.metadata.change_description = changes
    if paths is None:
        return []
    if not paths:
        raise NoChangesDetected()
    if args.metadata.change_description.targets is None:
        raise NoTestsForChanges()
    return args.metadata.change_description.targets