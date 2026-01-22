from typing import Dict, List, Tuple
import torio
def get_output_protocols() -> List[str]:
    """Get the supported output protocols.

    Returns:
        list of str: The names of supported output protocols

    Example
        >>> print(get_output_protocols())
        ... ['file', 'ftp', 'http', 'https', 'md5', 'pipe', 'prompeg', 'rtmp', 'tee', 'tcp', 'tls', 'udp', 'unix']
    """
    return ffmpeg_ext.get_output_protocols()