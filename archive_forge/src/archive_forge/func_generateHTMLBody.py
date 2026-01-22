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
def generateHTMLBody(mol, size, **kwargs):

    def toJson(x):
        return json.dumps(x, separators=(',', ':'))
    drawOptions = kwargs.get('drawOptions', _defaultDrawOptions)
    legend = kwargs.get('legend', None)
    useSVG = kwargs.get('useSVG', False)
    kekulize = Draw.shouldKekulize(mol, kwargs.get('kekulize', True))
    highlightAtoms = kwargs.get('highlightAtoms', []) or []
    highlightBonds = kwargs.get('highlightBonds', []) or []
    if not highlightAtoms and hasattr(mol, '__sssAtoms'):
        highlightAtoms = mol.__sssAtoms
        highlightBonds = [b.GetIdx() for b in mol.GetBonds() if b.GetBeginAtomIdx() in highlightAtoms and b.GetEndAtomIdx() in highlightAtoms]
    doc = minidom.Document()
    unique_id = str(uuid.uuid1())
    div = doc.createElement('div')
    for key, value in [('style', f'width: {size[0]}px; height: {size[1]}px; margin: auto;'), ('class', 'rdk-str-rnr-mol-container'), ('id', f'rdk-str-rnr-mol-{unique_id}'), ('data-mol', toDataMol(mol)), ('data-content', 'rdkit/molecule'), ('data-parent-node', parentNodeQuery)]:
        div.setAttribute(key, value)
    userDrawOpts = filterDefaultDrawOpts(drawOptions)
    molOpts = getOpts(mol)
    molOptsDashed = {}
    for key, value in molOpts.items():
        keyDashed = camelCaseOptToDataTag(key)
        if keyDashed == 'data-draw-opts':
            if isinstance(value, str):
                value = json.loads(value)
            if not isinstance(value, dict):
                raise ValueError(f'data-draw-opts: expected dict, found {str(type(value))}')
            userDrawOpts.update(value)
        else:
            molOptsDashed[keyDashed] = value
    if 'addAtomIndices' in userDrawOpts:
        addAtomIndices = userDrawOpts['addAtomIndices']
        if addAtomIndices:
            molOptsDashed['data-atom-idx'] = True
        userDrawOpts.pop('addAtomIndices')
    if highlightAtoms or highlightBonds:
        molOptsDashed['data-scaffold-highlight'] = True
        userDrawOpts['atoms'] = highlightAtoms
        userDrawOpts['bonds'] = highlightBonds
    if not kekulize:
        userDrawOpts['kekulize'] = False
    for key, value in molOptsDashed.items():
        if isinstance(value, Chem.Mol):
            value = toDataMol(value)
        elif not isinstance(value, str):
            value = toJson(value)
        div.setAttribute(key, value)
    if userDrawOpts:
        div.setAttribute('data-draw-opts', toJson(userDrawOpts))
    if useSVG:
        div.setAttribute('data-use-svg', 'true')
    if legend:
        outerTable = doc.createElement('table')
        outerTable.setAttribute('style', f'margin: auto;')
        molTr = doc.createElement('tr')
        molTd = doc.createElement('td')
        molTd.setAttribute('style', f'padding: 0;')
        molTd.appendChild(div)
        molTr.appendChild(molTd)
        nameTr = doc.createElement('tr')
        nameTh = doc.createElement('th')
        legendText = doc.createTextNode(legend)
        nameTh.appendChild(legendText)
        nameTh.setAttribute('style', f'text-align: center; background: white;')
        nameTr.appendChild(nameTh)
        outerTable.appendChild(molTr)
        outerTable.appendChild(nameTr)
        div = outerTable
    return xmlToNewline(div.toxml())