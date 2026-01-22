import sys
import logging
def mute(self):
    """
        Stop logging to stdout.
        """
    self.prev_level = self.streamHandler.level
    self.streamHandler.level = ERROR
    return self.prev_level