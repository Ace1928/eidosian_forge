from tornado.web import HTTPError
from traitlets.config.configurable import LoggingConfigurable
def rename_all_checkpoints(self, old_path, new_path):
    """Rename all checkpoints for old_path to new_path."""
    for cp in self.list_checkpoints(old_path):
        self.rename_checkpoint(cp['id'], old_path, new_path)