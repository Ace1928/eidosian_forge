import fixtures
from oslo_config import cfg
def set_config_dirs(self, config_dirs):
    """Specify a list of config dirs to read.

        This method allows you to predefine the list of configuration dirs
        that are loaded by oslo_config. It will ensure that your tests do not
        attempt to autodetect, and accidentally pick up config files from
        locally installed services.
        """
    if not isinstance(config_dirs, list):
        raise AttributeError('Please pass a list() to set_config_dirs()')
    if not self.conf._namespace:
        self.conf([])
    self.conf.default_config_dirs = config_dirs
    self.conf.reload_config_files()