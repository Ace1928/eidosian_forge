import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', type=str, help=_('ID or name of the instance.'))
@utils.arg('--directory', metavar='<directory>', type=str, help=_('Directory to write module content files in. It will be created if it does not exist. Defaults to the current directory.'))
@utils.arg('--prefix', metavar='<filename_prefix>', type=str, help=_('Prefix to prepend to generated filename for each module.'))
@utils.service_type('database')
def do_module_retrieve(cs, args):
    """Retrieve module contents from an instance."""
    instance = _find_instance(cs, args.instance)
    saved_modules = cs.instances.module_retrieve(instance, args.directory, args.prefix)
    for module_name, filename in saved_modules.items():
        print(_("Module contents for '%(module)s' written to '%(file)s'") % {'module': module_name, 'file': filename})