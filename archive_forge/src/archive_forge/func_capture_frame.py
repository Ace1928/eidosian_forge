import json
import os
import os.path
import tempfile
from typing import List, Optional
from gym import error, logger
def capture_frame(self):
    """Render the given `env` and add the resulting frame to the video."""
    frame = self.env.render()
    if isinstance(frame, List):
        self.render_history += frame
        frame = frame[-1]
    if not self.functional:
        return
    if self._closed:
        logger.warn('The video recorder has been closed and no frames will be captured anymore.')
        return
    logger.debug('Capturing video frame: path=%s', self.path)
    if frame is None:
        if self._async:
            return
        else:
            logger.warn(f'Env returned None on `render()`. Disabling further rendering for video recorder by marking as disabled: path={self.path} metadata_path={self.metadata_path}')
            self.broken = True
    else:
        self.recorded_frames.append(frame)