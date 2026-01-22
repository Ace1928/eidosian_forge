@property
def had_inconsistency(self):
    return self.direct_inconsistency or self.bases_had_inconsistency