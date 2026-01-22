import sys
from itertools import combinations
from rdkit import Chem, DataStructs
from rdkit.Chem import rdqueries
def atomContrib(subs, mol, tverskyThresh=0.8):
    """ atomContrib algorithm
  generate fp of query_substructs (qfp)

  loop through atoms of smiles
    For each atom
    Generate partial fp of the atom (pfp)
    Find Tversky sim of pfp in qfp
    If Tversky < 0.8, mark atom in smiles

  Loop through marked atoms
    If marked atom in ring - turn all atoms in that ring to * (aromatic) or Sc (aliphatic)
    For each marked atom
      If aromatic turn to a *
      If aliphatic turn to a Sc

  Return modified smiles
  """

    def partialSimilarity(atomID):
        """ Determine similarity for the atoms set by atomID """
        modifiedFP = DataStructs.ExplicitBitVect(1024)
        modifiedFP.SetBitsFromList(aBits[atomID])
        return DataStructs.TverskySimilarity(subsFp, modifiedFP, 0, 1)
    pMol = Chem.Mol(mol)
    aBits = []
    _ = Chem.RDKFingerprint(pMol, atomBits=aBits, **rdkitFpParams)
    qsMol = Chem.MolFromSmiles(subs)
    subsFp = Chem.RDKFingerprint(qsMol, **rdkitFpParams)
    marked = set()
    for atom in pMol.GetAtoms():
        atomIdx = atom.GetIdx()
        if partialSimilarity(atomIdx) < tverskyThresh:
            marked.add(atomIdx)
    markRingAtoms = set()
    for ring in pMol.GetRingInfo().AtomRings():
        if any((ringAtom in marked for ringAtom in ring)):
            markRingAtoms.update(ring)
    marked.update(markRingAtoms)
    if marked:
        for idx in marked:
            if pMol.GetAtomWithIdx(idx).GetIsAromatic():
                pMol.GetAtomWithIdx(idx).SetAtomicNum(0)
                pMol.GetAtomWithIdx(idx).SetNoImplicit(True)
            else:
                pMol.GetAtomWithIdx(idx).SetAtomicNum(21)
        try:
            Chem.SanitizeMol(pMol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE ^ Chem.SANITIZE_SETAROMATICITY)
        except Exception:
            sys.stderr.write(f"Can't parse smiles: {Chem.MolToSmiles(pMol)}\n")
            pMol = Chem.Mol(mol)
    return pMol