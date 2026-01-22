import numpy
from rdkit import Geometry
from rdkit.Chem import ChemicalFeatures
from rdkit.RDLogger import logger
def initFromLines(self, lines):
    import re
    spaces = re.compile('[ \t]+')
    feats = []
    rads = []
    for lineNum, line in enumerate(lines):
        txt = line.split('#')[0].strip()
        if txt:
            splitL = spaces.split(txt)
            if len(splitL) < 5:
                logger.error(f'Input line {lineNum} only contains {len(splitL)} fields, 5 are required. Read failed.')
                return
            fName = splitL[0]
            try:
                xP = float(splitL[1])
                yP = float(splitL[2])
                zP = float(splitL[3])
                rad = float(splitL[4])
            except ValueError:
                logger.error(f'Error parsing a number of line {lineNum}. Read failed.')
                return
            feats.append(ChemicalFeatures.FreeChemicalFeature(fName, fName, Geometry.Point3D(xP, yP, zP)))
            rads.append(rad)
    self._initializeFeats(feats, rads)