import inspect
import re
import six
def ui_eval_param(self, ui_value, type, default):
    """
        Evaluates a user-provided parameter value using a given type helper.
        If the parameter value is None, the default will be returned. If the
        ui_value does not check out with the type helper, and execution error
        will be raised.

        @param ui_value: The user provided parameter value.
        @type ui_value: str
        @param type: The ui_type to be used
        @type type: str
        @param default: The default value to return.
        @type default: any
        @return: The evaluated parameter value.
        @rtype: depends on type
        @raise ExecutionError: If evaluation fails.
        """
    type_method = self.get_type_method(type)
    if ui_value is None:
        return default
    else:
        try:
            value = type_method(ui_value)
        except ValueError as msg:
            raise ExecutionError(msg)
        else:
            return value