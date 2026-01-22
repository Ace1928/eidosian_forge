import os
import argparse
import inspect
import sys
from ..interfaces.base import Interface, InputMultiPath, traits
from ..interfaces.base.support import get_trait_desc
from .misc import str2bool
def run_instance(interface, options):
    print('setting function inputs')
    for input_name, _ in list(interface.inputs.items()):
        if getattr(options, input_name) is not None:
            value = getattr(options, input_name)
            try:
                setattr(interface.inputs, input_name, value)
            except ValueError as e:
                print("Error when setting the value of %s: '%s'" % (input_name, str(e)))
    print(interface.inputs)
    res = interface.run()
    print(res.outputs)