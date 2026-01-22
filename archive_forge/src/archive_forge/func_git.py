from typing import Any
from wandb.sdk.internal.internal_api import Api as InternalApi
@property
def git(self):
    return self.api.git