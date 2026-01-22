from dataclasses import asdict, dataclass, field
from typing import Type
from typing_extensions import override
def reset_on_run(self) -> None:
    self.optimizer.reset_on_run()