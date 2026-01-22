from ._LinearDrawer import LinearDrawer
from ._CircularDrawer import CircularDrawer
from ._Track import Track
from Bio.Graphics import _write
def set_all_tracks(self, attr, value):
    """Set the passed attribute of all tracks in the set to the passed value.

        Arguments:
         - attr    - An attribute of the Track class.
         - value   - The value to set that attribute.

        set_all_tracks(self, attr, value)
        """
    for track in self.tracks.values():
        if hasattr(track, attr):
            setattr(track, attr, value)