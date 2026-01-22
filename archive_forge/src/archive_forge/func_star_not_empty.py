import os
import re
def star_not_empty(self, pattern):
    """
        Ensure that * will not match an empty segment.
        """

    def handle_segment(match):
        segment = match.group(0)
        return '?*' if segment == '*' else segment
    not_seps_pattern = f'[^{re.escape(self.seps)}]+'
    return re.sub(not_seps_pattern, handle_segment, pattern)