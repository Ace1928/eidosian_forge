import os
from typing import Callable, Optional
import gym
from gym import logger
from gym.wrappers.monitoring import video_recorder
def close_video_recorder(self):
    """Closes the video recorder if currently recording."""
    if self.recording:
        assert self.video_recorder is not None
        self.video_recorder.close()
    self.recording = False
    self.recorded_frames = 1