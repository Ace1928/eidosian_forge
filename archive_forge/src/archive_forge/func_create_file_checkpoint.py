from tornado.web import HTTPError
from traitlets.config.configurable import LoggingConfigurable
def create_file_checkpoint(self, content, format, path):
    """Create a checkpoint of the current state of a file

        Returns a checkpoint model for the new checkpoint.
        """
    raise NotImplementedError