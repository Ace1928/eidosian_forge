import base64
import json
import logging
import re
import uuid
from xml.dom import minidom
from IPython.display import HTML, display
from rdkit import Chem
from rdkit.Chem import Draw
from . import rdMolDraw2D
def filterDefaultDrawOpts(molDrawOptions):
    global _defaultDrawOptionsDict
    if not isinstance(molDrawOptions, rdMolDraw2D.MolDrawOptions):
        raise ValueError(f'Bad args ({str(type(molDrawOptions))}) for {__name__}.filterDefaultDrawOpts(molDrawOptions: Chem.Draw.rdMolDraw2D.MolDrawOptions)')
    molDrawOptionsAsDict = _molDrawOptionsToDict(molDrawOptions)
    if _defaultDrawOptionsDict is None:
        _defaultDrawOptionsDict = _molDrawOptionsToDict()
    return {key: value for key, value in molDrawOptionsAsDict.items() if not _isDrawOptionEqual(value, _defaultDrawOptionsDict[key])}