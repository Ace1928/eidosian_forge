from typing import List, Optional
from torchaudio._internal.module_utils import deprecated
from . import utils
from .common import AudioMetaData
@deprecated('With dispatcher enabled, this function is no-op. You can remove the function call.')
def set_audio_backend(backend: Optional[str]):
    """Set the global backend.

    This is a no-op when dispatcher mode is enabled.

    Args:
        backend (str or None): Name of the backend.
            One of ``"sox_io"`` or ``"soundfile"`` based on availability
            of the system. If ``None`` is provided the  current backend is unassigned.
    """
    pass