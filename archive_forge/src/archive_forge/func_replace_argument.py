from copy import deepcopy
from flask import current_app, request
from werkzeug.datastructures import MultiDict, FileStorage
from werkzeug import exceptions
import flask_restful
import decimal
import six
def replace_argument(self, name, *args, **kwargs):
    """ Replace the argument matching the given name with a new version. """
    new_arg = self.argument_class(name, *args, **kwargs)
    for index, arg in enumerate(self.args[:]):
        if new_arg.name == arg.name:
            del self.args[index]
            self.args.append(new_arg)
            break
    return self