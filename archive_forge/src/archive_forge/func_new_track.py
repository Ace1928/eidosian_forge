from ._LinearDrawer import LinearDrawer
from ._CircularDrawer import CircularDrawer
from ._Track import Track
from Bio.Graphics import _write
def new_track(self, track_level, **args):
    """Add a new Track to the diagram at a given level.

        The track is returned for further user manipulation.

        Arguments:
            - track_level   - an integer. The level at which the track will be
              drawn (above an arbitrary baseline).

        new_track(self, track_level)
        """
    newtrack = Track()
    for key in args:
        setattr(newtrack, key, args[key])
    if track_level not in self.tracks:
        self.tracks[track_level] = newtrack
    else:
        occupied_levels = sorted(self.get_levels())
        occupied_levels.reverse()
        for val in occupied_levels:
            if val >= track_level:
                self.tracks[val + 1] = self.tracks[val]
        self.tracks[track_level] = newtrack
    self.tracks[track_level].track_level = track_level
    return newtrack