from io import BytesIO
from . import osutils, progress, trace
from .i18n import gettext
from .ui import ui_factory
@classmethod
def guess_renames(klass, from_tree, to_tree, dry_run=False):
    """Guess which files to rename, and perform the rename.

        We assume that unversioned files and missing files indicate that
        versioned files have been renamed outside of Bazaar.

        :param from_tree: A tree to compare from
        :param to_tree: A write-locked working tree.
        """
    required_parents = {}
    with ui_factory.nested_progress_bar() as task:
        pp = progress.ProgressPhase('Guessing renames', 4, task)
        with from_tree.lock_read():
            rn = klass(to_tree)
            pp.next_phase()
            missing_files, missing_parents, candidate_files = rn._find_missing_files(from_tree)
            pp.next_phase()
            rn.add_file_edge_hashes(from_tree, missing_files)
        pp.next_phase()
        matches = rn.file_match(candidate_files)
        parents_matches = matches
        while len(parents_matches) > 0:
            required_parents = rn.get_required_parents(parents_matches)
            parents_matches = rn.match_parents(required_parents, missing_parents)
            matches.update(parents_matches)
        pp.next_phase()
        delta = rn._make_inventory_delta(matches)
        for old, new, file_id, entry in delta:
            trace.note(gettext('{0} => {1}').format(old, new))
        if not dry_run:
            to_tree.add(required_parents)
            to_tree.apply_inventory_delta(delta)