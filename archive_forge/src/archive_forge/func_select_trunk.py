from operator import itemgetter
from breezy import controldir
from ... import errors, osutils, transport
from ...trace import note, show_error
from .helpers import best_format_for_objects_in_a_repository, single_plural
def select_trunk(self, ref_names):
    """Given a set of ref names, choose one as the trunk."""
    for candidate in ['refs/heads/master']:
        if candidate in ref_names:
            return candidate
    return self.last_ref