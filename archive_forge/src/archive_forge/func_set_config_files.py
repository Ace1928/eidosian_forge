import fixtures
from oslo_config import cfg
def set_config_files(self, config_files):
    """Specify a list of config files to read.

        This method allows you to predefine the list of configuration files
        that are loaded by oslo_config. It will ensure that your tests do not
        attempt to autodetect, and accidentally pick up config files from
        locally installed services.
        """
    if not isinstance(config_files, list):
        raise AttributeError('Please pass a list() to set_config_files()')
    if not self.conf._namespace:
        self.conf.__call__(args=[])
    self.conf.default_config_files = config_files
    self.conf.reload_config_files()