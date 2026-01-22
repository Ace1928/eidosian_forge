from typing import Dict, List
import torchaudio
def list_read_formats() -> List[str]:
    """List the supported audio formats for read

    Returns:
        List[str]: List of supported audio formats
    """
    return sox_ext.list_read_formats()