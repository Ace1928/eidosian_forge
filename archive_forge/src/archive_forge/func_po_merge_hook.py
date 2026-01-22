from ... import \
from ... import version_info  # noqa: F401
from ...hooks import install_lazy_named_hook
def po_merge_hook(merger):
    """Merger.merge_file_content hook for po files."""
    from .po_merge import PoMerger
    return PoMerger(merger)