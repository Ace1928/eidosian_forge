from io import BytesIO
from . import osutils, progress, trace
from .i18n import gettext
from .ui import ui_factory
def get_all_hits(self, paths):
    """Find all the hit counts for the listed paths in the tree.

        :return: A list of tuples of count, path, file_id.
        """
    all_hits = []
    with ui_factory.nested_progress_bar() as task:
        for num, path in enumerate(paths):
            task.update(gettext('Determining hash hits'), num, len(paths))
            hits = self.hitcounts(self.tree.get_file_lines(path))
            all_hits.extend(((v, path, k) for k, v in hits.items()))
    return all_hits