import inspect
import re
import six
def get_command_syntax(self, command):
    """
        @return: A list of formatted syntax descriptions for the command:
            - (syntax, comments, default_values)
            - syntax is the syntax definition line.
            - comments is a list of additionnal comments about the syntax.
            - default_values is a string with the default parameters values.
        @rtype: (str, [str...], str)
        @param command: The command to document.
        @type command: str
        """
    method = self.get_command_method(command)
    spec = inspect.getfullargspec(method)
    parameters = spec.args[1:]
    if spec.defaults is None:
        num_defaults = 0
    else:
        num_defaults = len(spec.defaults)
    if num_defaults != 0:
        required_parameters = parameters[:-num_defaults]
        optional_parameters = parameters[-num_defaults:]
    else:
        required_parameters = parameters
        optional_parameters = []
    self.shell.log.debug('Required: %s' % str(required_parameters))
    self.shell.log.debug('Optional: %s' % str(optional_parameters))
    syntax = '%s ' % command
    required_parameters_str = ''
    for param in required_parameters:
        required_parameters_str += '%s ' % param
    syntax += required_parameters_str
    optional_parameters_str = ''
    for param in optional_parameters:
        optional_parameters_str += '[%s] ' % param
    syntax += optional_parameters_str
    comments = []
    if spec.varargs is not None:
        syntax += '[%s...] ' % spec.varargs
    if spec.varkw is not None:
        syntax += '[%s=value...] ' % spec.varkw
    default_values = ''
    if num_defaults > 0:
        for index, param in enumerate(optional_parameters):
            if spec.defaults[index] is not None:
                default_values += '%s=%s ' % (param, str(spec.defaults[index]))
    return (syntax, comments, default_values)