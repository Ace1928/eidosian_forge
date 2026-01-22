import abc
@abc.abstractmethod
def open_source_from_opt_group(self, conf, group_name):
    """Return an open configuration source.

        Uses group_name to find the configuration settings for the new
        source and then open the configuration source and return it.

        If a source cannot be open, raises an appropriate exception.

        :param conf: The active configuration option handler from which
                     to seek configuration values.
        :type conf: ConfigOpts
        :param group_name: The configuration option group name where the
                           options for the source are stored.
        :type group_name: str
        :returns: an instance of a subclass of ConfigurationSource
        """