from ... import version_info  # noqa: F401
from ...hooks import install_lazy_named_hook
def news_merge_hook(merger):
    """Merger.merge_file_content hook for bzr-format NEWS files."""
    from .news_merge import NewsMerger
    return NewsMerger(merger)