import argparse
import sys
from rdkit import Chem, Geometry
from rdkit.Chem import rdDepictor
def processArgs(args):
    patt = args.patt
    if patt:
        patt = Chem.MolFromSmarts(patt)
    if args.useSmiles:
        core = Chem.MolFromSmiles(args.core)
        mol = Chem.MolFromSmiles(args.mol)
        rdDepictor.Compute2DCoords(core)
    else:
        core = Chem.MolFromMolFile(args.core)
        mol = Chem.MolFromMolFile(args.mol)
    AlignDepict(mol, core, patt)
    print(Chem.MolToMolBlock(mol), file=args.outF)