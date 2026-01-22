from typing import Dict, List
import torchaudio
def set_use_threads(use_threads: bool):
    """Set multithread option for sox effect chain

    Args:
        use_threads (bool): When ``True``, enables ``libsox``'s parallel effects channels processing.
            To use mutlithread, the underlying ``libsox`` has to be compiled with OpenMP support.

    See Also:
        http://sox.sourceforge.net/sox.html
    """
    sox_ext.set_use_threads(use_threads)