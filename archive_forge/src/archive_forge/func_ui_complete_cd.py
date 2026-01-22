import inspect
import re
import six
def ui_complete_cd(self, parameters, text, current_param):
    """
        Parameter auto-completion method for user command cd.
        @param parameters: Parameters on the command line.
        @type parameters: dict
        @param text: Current text of parameter being typed by the user.
        @type text: str
        @param current_param: Name of parameter to complete.
        @type current_param: str
        @return: Possible completions
        @rtype: list of str
        """
    if current_param == 'path':
        completions = self.ui_complete_ls(parameters, text, current_param)
        completions.extend([nav for nav in ['<', '>'] if nav.startswith(text)])
        return completions