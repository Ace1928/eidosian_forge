import inspect
import re
import six
def get_command_signature(self, command):
    """
        Get a command's signature.
        @param command: The command to get the signature of.
        @type command: str
        @return: (parameters, free_pparams, free_kparams) where parameters is a
        list of all the command's parameters and free_pparams and free_kparams
        booleans set to True is the command accepts an arbitrary number of,
        respectively, pparams and kparams.
        @rtype: ([str...], bool, bool)
        """
    method = self.get_command_method(command)
    spec = inspect.getfullargspec(method)
    parameters = spec.args[1:]
    if spec.varargs is not None:
        free_pparams = spec.varargs
    else:
        free_pparams = False
    if spec.varkw is not None:
        free_kparams = spec.varkw
    else:
        free_kparams = False
    self.shell.log.debug('Signature is %s, %s, %s.' % (str(parameters), str(free_pparams), str(free_kparams)))
    return (parameters, free_pparams, free_kparams)