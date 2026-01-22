from dataclasses import asdict, dataclass, field
from typing import Type
from typing_extensions import override
def increment_processed(self) -> None:
    if not isinstance(self.total, _ProcessedTracker):
        raise TypeError(f"`{self.total.__class__.__name__}` doesn't have a `processed` attribute")
    self.total.processed += 1
    self.current.processed += 1