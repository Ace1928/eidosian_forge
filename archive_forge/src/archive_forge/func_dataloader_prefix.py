from dataclasses import dataclass
from typing import Optional
from pytorch_lightning.utilities.enums import LightningEnum
@property
def dataloader_prefix(self) -> Optional[str]:
    if self in (self.VALIDATING, self.SANITY_CHECKING):
        return 'val'
    return self.value