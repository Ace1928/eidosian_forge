import math
def call_desc(self, mol, index):
    if hasattr(mol, self.func_key):
        results = getattr(mol, self.func_key, None)
        if results is not None:
            return results[index]
    try:
        results = self.func(mol)
    except Exception:
        return math.nan
    setattr(mol, self.func_key, results)
    return results[index]