import inspect
import re
import six
def get_command_method(self, command):
    """
        @param command: The command to get the method for.
        @type command: str
        @return: The user command support method.
        @rtype: method
        @raise ValueError: If the command is not found.
        """
    prefix = self.ui_command_method_prefix
    if command in self.list_commands():
        return getattr(self, '%s%s' % (prefix, command))
    else:
        self.shell.log.debug('No command named %s in %s (%s)' % (command, self.name, self.path))
        raise ValueError('No command named "%s".' % command)