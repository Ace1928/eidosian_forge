from typing import Any, Dict, List, Optional, Tuple
import grpc
from ray.util.annotations import DeveloperAPI, PublicAPI
def invocation_metadata(self) -> List[Tuple[str, str]]:
    """Accesses the metadata sent by the client.

        Returns:
          The invocation :term:`metadata`.
        """
    return self._invocation_metadata