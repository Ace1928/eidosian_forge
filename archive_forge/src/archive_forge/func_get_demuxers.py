from typing import Dict, List, Tuple
import torio
def get_demuxers() -> Dict[str, str]:
    """Get the available demuxers.

    Returns:
        Dict[str, str]: Mapping from demuxer (format) short name to long name.

    Example
        >>> for k, v in get_demuxers().items():
        >>>     print(f"{k}: {v}")
        ... aa: Audible AA format files
        ... aac: raw ADTS AAC (Advanced Audio Coding)
        ... aax: CRI AAX
        ... ac3: raw AC-3
    """
    return ffmpeg_ext.get_demuxers()