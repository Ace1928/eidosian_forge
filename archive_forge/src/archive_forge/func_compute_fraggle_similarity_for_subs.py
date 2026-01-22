import sys
from itertools import combinations
from rdkit import Chem, DataStructs
from rdkit.Chem import rdqueries
def compute_fraggle_similarity_for_subs(inMol, qMol, qSmi, qSubs, tverskyThresh=0.8):
    qFP = Chem.RDKFingerprint(qMol, **rdkitFpParams)
    iFP = Chem.RDKFingerprint(inMol, **rdkitFpParams)
    rdkit_sim = DataStructs.TanimotoSimilarity(qFP, iFP)
    qm_key = f'{qSubs}_{qSmi}'
    if qm_key in modified_query_fps:
        qmMolFp = modified_query_fps[qm_key]
    else:
        qmMol = atomContrib(qSubs, qMol, tverskyThresh)
        qmMolFp = Chem.RDKFingerprint(qmMol, **rdkitFpParams)
        modified_query_fps[qm_key] = qmMolFp
    rmMol = atomContrib(qSubs, inMol, tverskyThresh)
    try:
        rmMolFp = Chem.RDKFingerprint(rmMol, **rdkitFpParams)
        fraggle_sim = max(DataStructs.FingerprintSimilarity(qmMolFp, rmMolFp), rdkit_sim)
    except Exception:
        sys.stderr.write(f"Can't generate fp for: {Chem.MolToSmiles(rmMol)}\n")
        fraggle_sim = 0.0
    return (rdkit_sim, fraggle_sim)