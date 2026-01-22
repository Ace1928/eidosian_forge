from twisted.plugin import getPlugins
from twisted.trial import unittest
from twisted.trial.itrial import IReporter
def getPluginsByLongOption(self, longOption):
    """
        Return the Trial reporter plugin with the given long option.

        If more than one is found, raise ValueError. If none are found, raise
        IndexError.
        """
    plugins = [plugin for plugin in getPlugins(IReporter) if plugin.longOpt == longOption]
    if len(plugins) > 1:
        raise ValueError('More than one plugin found with long option %r: %r' % (longOption, plugins))
    return plugins[0]