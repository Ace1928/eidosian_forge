import time
from . import debug, errors, osutils, revision, trace
def find_lefthand_merger(self, merged_key, tip_key):
    """Find the first lefthand ancestor of tip_key that merged merged_key.

        We do this by first finding the descendants of merged_key, then
        walking through the lefthand ancestry of tip_key until we find a key
        that doesn't descend from merged_key.  Its child is the key that
        merged merged_key.

        :return: The first lefthand ancestor of tip_key to merge merged_key.
            merged_key if it is a lefthand ancestor of tip_key.
            None if no ancestor of tip_key merged merged_key.
        """
    descendants = self.find_descendants(merged_key, tip_key)
    candidate_iterator = self.iter_lefthand_ancestry(tip_key)
    last_candidate = None
    for candidate in candidate_iterator:
        if candidate not in descendants:
            return last_candidate
        last_candidate = candidate