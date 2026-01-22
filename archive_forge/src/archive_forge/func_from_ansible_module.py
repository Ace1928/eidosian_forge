from __future__ import absolute_import, division, print_function
import os
@classmethod
def from_ansible_module(cls, module):
    return cls.from_dict(module.params)