import logging
import sys
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
from wandb.sdk.artifacts.artifact import Artifact
def used_artifacts(self) -> Optional[Iterable[Artifact]]:
    ...