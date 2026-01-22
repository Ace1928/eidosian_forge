import sys
from PyQt5 import QtWidgets
from DAWModules import SoundModule
from PyQt5.QtCore import Qt
import pyaudio
import numpy as np
from typing import Dict, Type, List, Tuple
def audio_callback(self, in_data, frame_count, time_info, status) -> Tuple[bytes, int]:
    """
        Processes audio in real-time by passing it through each active module sequentially.

        Parameters:
            in_data (bytes): Input audio data (not used in this callback).
            frame_count (int): The number of frames to process.
            time_info (dict): Timing information.
            status (int): Stream status.

        Returns:
            Tuple[bytes, int]: A tuple containing the processed audio data and a flag indicating whether to continue the stream.
        """
    data = np.zeros(frame_count, dtype=np.float32)
    for module in self.modules.values():
        data = module.process_sound(data)
    return (data.tobytes(), pyaudio.paContinue)