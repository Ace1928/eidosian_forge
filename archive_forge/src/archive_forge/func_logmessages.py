from warnings import warn
import logging
import sys
from rdkit import Chem
from .errors import StopValidateError
from .validations import VALIDATIONS
@property
def logmessages(self):
    return [self.format(record) for record in self.logs]