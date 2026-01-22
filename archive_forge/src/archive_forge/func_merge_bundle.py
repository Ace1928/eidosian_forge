from ... import ui
from ...i18n import gettext
from ...merge import Merger
from ...progress import ProgressPhase
from ...trace import note
from ..vf_repository import install_revision
def merge_bundle(reader, tree, check_clean, merge_type, reprocess, show_base, change_reporter=None):
    """Merge a revision bundle into the current tree."""
    with ui.ui_factory.nested_progress_bar() as pb:
        pp = ProgressPhase('Merge phase', 6, pb)
        pp.next_phase()
        install_bundle(tree.branch.repository, reader)
        merger = Merger(tree.branch, this_tree=tree, change_reporter=change_reporter)
        merger.pp = pp
        merger.pp.next_phase()
        if check_clean and tree.has_changes():
            raise errors.UncommittedChanges(self)
        merger.other_rev_id = reader.target
        merger.other_tree = merger.revision_tree(reader.target)
        merger.other_basis = reader.target
        merger.pp.next_phase()
        merger.find_base()
        if merger.base_rev_id == merger.other_rev_id:
            note(gettext('Nothing to do.'))
            return 0
        merger.merge_type = merge_type
        merger.show_base = show_base
        merger.reprocess = reprocess
        conflicts = merger.do_merge()
        merger.set_pending()
    return conflicts