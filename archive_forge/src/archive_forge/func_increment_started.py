from dataclasses import asdict, dataclass, field
from typing import Type
from typing_extensions import override
def increment_started(self) -> None:
    if not isinstance(self.total, _StartedTracker):
        raise TypeError(f"`{self.total.__class__.__name__}` doesn't have a `started` attribute")
    self.total.started += 1
    self.current.started += 1