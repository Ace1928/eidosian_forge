import base64
import copy
import html
import warnings
from io import BytesIO
import IPython
from IPython.display import HTML, SVG
from rdkit import Chem
from rdkit.Chem import Draw, rdchem, rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D
from . import InteractiveRenderer
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from IPython import display
def drawMol3D(m, view=None, confId=-1, drawAs=None, bgColor=None, size=None):
    if bgColor is None:
        bgColor = bgcolor_3d
    if size is None:
        size = molSize_3d
    if view is None:
        view = py3Dmol.view(width=size[0], height=size[1])
    view.removeAllModels()
    try:
        ms = iter(m)
        for m in ms:
            addMolToView(m, view, confId, drawAs)
    except TypeError:
        addMolToView(m, view, confId, drawAs)
    view.setBackgroundColor(bgColor)
    view.zoomTo()
    return view.show()