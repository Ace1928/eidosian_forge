import os
from troveclient import base
from troveclient import common
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules as core_modules
from swiftclient import client as swift_client
def module_retrieve(self, instance, directory=None, prefix=None):
    """Retrieve the module data file from an instance.  This includes
        the contents of the module data file.
        """
    if directory:
        try:
            os.makedirs(directory, exist_ok=True)
        except TypeError:
            try:
                os.makedirs(directory)
            except OSError:
                if not os.path.isdir(directory):
                    raise
    else:
        directory = '.'
    prefix = prefix or ''
    if prefix and (not prefix.endswith('_')):
        prefix += '_'
    module_list = self._modules_get(instance, from_guest=True, include_contents=True)
    saved_modules = {}
    for module in module_list:
        filename = '%s%s_%s_%s.dat' % (prefix, module.name, module.datastore, module.datastore_version)
        full_filename = os.path.expanduser(os.path.join(directory, filename))
        with open(full_filename, 'wb') as fh:
            fh.write(utils.decode_data(module.contents))
        saved_modules[module.name] = full_filename
    return saved_modules