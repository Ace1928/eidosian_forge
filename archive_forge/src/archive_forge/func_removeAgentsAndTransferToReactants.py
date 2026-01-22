import copy
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
def removeAgentsAndTransferToReactants(rxn):
    tmp = []
    rxn.RemoveAgentTemplates(tmp)
    for a in tmp:
        rxn.AddReactantTemplate(a)