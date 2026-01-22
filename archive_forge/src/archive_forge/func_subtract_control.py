import warnings
import json
import csv
import numpy as np
from Bio import BiopythonParserWarning
def subtract_control(self, control='A01', wells=None):
    """Subtract a 'control' well from the other plates wells.

        By default the control is subtracted to all wells, unless
        a list of well ID is provided

        The control well should belong to the plate
        A new PlateRecord object is returned
        """
    if control not in self:
        raise ValueError('Control well not present in plate')
    wcontrol = self[control]
    if wells is None:
        wells = self._wells.keys()
    missing = {w for w in wells if w not in self}
    if missing:
        raise ValueError('Some wells to be subtracted are not present')
    nwells = []
    for w in self:
        if w.id in wells:
            nwells.append(w - wcontrol)
        else:
            nwells.append(w)
    newp = PlateRecord(self.id, wells=nwells)
    return newp