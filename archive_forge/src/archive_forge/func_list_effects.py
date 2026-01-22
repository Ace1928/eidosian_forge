from typing import Dict, List
import torchaudio
def list_effects() -> Dict[str, str]:
    """List the available sox effect names

    Returns:
        Dict[str, str]: Mapping from ``effect name`` to ``usage``
    """
    return dict(sox_ext.list_effects())