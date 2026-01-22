import numpy as np
import numba
def iter(self, epoch):
    all_batches, all_totseqs, all_curseqs = self.generate_batches(epoch, set_stats=True)
    for batch, totseq, curseq in zip(all_batches, all_totseqs, all_curseqs):
        yield (batch, totseq, curseq)