import enum
import logging
import os
import types
import typing
@property
def callback_def(self) -> str:
    return '{} ({})'.format(self.rtype, ' ,'.join(self.arguments))