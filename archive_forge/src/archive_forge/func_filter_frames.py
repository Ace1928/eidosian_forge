import re
import html
from paste.util import PySourceColor
def filter_frames(self, frames):
    """
        Removes any frames that should be hidden, according to the
        values of traceback_hide, self.show_hidden_frames, and the
        hidden status of the final frame.
        """
    if self.show_hidden_frames:
        return frames
    new_frames = []
    hidden = False
    for frame in frames:
        hide = frame.traceback_hide
        if hide == 'before':
            new_frames = []
            hidden = False
        elif hide == 'before_and_this':
            new_frames = []
            hidden = False
            continue
        elif hide == 'reset':
            hidden = False
        elif hide == 'reset_and_this':
            hidden = False
            continue
        elif hide == 'after':
            hidden = True
        elif hide == 'after_and_this':
            hidden = True
            continue
        elif hide:
            continue
        elif hidden:
            continue
        new_frames.append(frame)
    if frames[-1] not in new_frames:
        return frames
    return new_frames