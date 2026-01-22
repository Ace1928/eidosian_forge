from functools import reduce
import copy
import math
import random
import sys
import warnings
from Bio import File
from Bio.Data import IUPACData
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning, BiopythonWarning
from Bio.Nexus.StandardData import StandardData
from Bio.Nexus.Trees import Tree
def write_nexus_data(self, filename=None, matrix=None, exclude=(), delete=(), blocksize=None, interleave=False, interleave_by_partition=False, comment=None, omit_NEXUS=False, append_sets=True, mrbayes=False, codons_block=True):
    """Write a nexus file with data and sets block to a file or handle.

        Character sets and partitions are appended by default, and are
        adjusted according to excluded characters (i.e. character sets
        still point to the same sites (not necessarily same positions),
        without including the deleted characters.

        - filename - Either a filename as a string (which will be opened,
          written to and closed), or a handle object (which will
          be written to but NOT closed).
        - interleave_by_partition - Optional name of partition (string)
        - omit_NEXUS - Boolean.  If true, the '#NEXUS' line normally at the
          start of the file is omitted.

        Returns the filename/handle used to write the data.
        """
    if not matrix:
        matrix = self.matrix
    if not matrix:
        return
    if not filename:
        filename = self.filename
    if [t for t in delete if not self._check_taxlabels(t)]:
        raise NexusError('Unknown taxa: %s' % ', '.join(set(delete).difference(set(self.taxlabels))))
    if interleave_by_partition:
        if interleave_by_partition not in self.charpartitions:
            raise NexusError(f'Unknown partition: {interleave_by_partition!r}')
        else:
            partition = self.charpartitions[interleave_by_partition]
            names = _sort_keys_by_values(partition)
            newpartition = {}
            for p in partition:
                newpartition[p] = [c for c in partition[p] if c not in exclude]
    undelete = [taxon for taxon in self.taxlabels if taxon in matrix and taxon not in delete]
    cropped_matrix = _seqmatrix2strmatrix(self.crop_matrix(matrix, exclude=exclude, delete=delete))
    ntax_adjusted = len(undelete)
    nchar_adjusted = len(cropped_matrix[undelete[0]])
    if not undelete or (undelete and undelete[0] == ''):
        return
    with File.as_handle(filename, mode='w') as fh:
        if not omit_NEXUS:
            fh.write('#NEXUS\n')
        if comment:
            fh.write('[' + comment + ']\n')
        fh.write('begin data;\n')
        fh.write('dimensions ntax=%d nchar=%d;\n' % (ntax_adjusted, nchar_adjusted))
        fh.write('format datatype=' + self.datatype)
        if self.respectcase:
            fh.write(' respectcase')
        if self.missing:
            fh.write(' missing=' + self.missing)
        if self.gap:
            fh.write(' gap=' + self.gap)
        if self.matchchar:
            fh.write(' matchchar=' + self.matchchar)
        if self.labels:
            fh.write(' labels=' + self.labels)
        if self.equate:
            fh.write(' equate=' + self.equate)
        if interleave or interleave_by_partition:
            fh.write(' interleave')
        fh.write(';\n')
        if self.charlabels:
            newcharlabels = self._adjust_charlabels(exclude=exclude)
            clkeys = sorted(newcharlabels)
            fh.write('charlabels ' + ', '.join((f'{k + 1} {safename(newcharlabels[k])}' for k in clkeys)) + ';\n')
        fh.write('matrix\n')
        if not blocksize:
            if interleave:
                blocksize = 70
            else:
                blocksize = self.nchar
        namelength = max((len(safename(t, mrbayes=mrbayes)) for t in undelete))
        if interleave_by_partition:
            seek = 0
            for p in names:
                fh.write(f'[{interleave_by_partition}: {p}]\n')
                if len(newpartition[p]) > 0:
                    for taxon in undelete:
                        fh.write(safename(taxon, mrbayes=mrbayes).ljust(namelength + 1))
                        fh.write(cropped_matrix[taxon][seek:seek + len(newpartition[p])] + '\n')
                    fh.write('\n')
                else:
                    fh.write('[empty]\n\n')
                seek += len(newpartition[p])
        elif interleave:
            for seek in range(0, nchar_adjusted, blocksize):
                for taxon in undelete:
                    fh.write(safename(taxon, mrbayes=mrbayes).ljust(namelength + 1))
                    fh.write(cropped_matrix[taxon][seek:seek + blocksize] + '\n')
                fh.write('\n')
        else:
            for taxon in undelete:
                if blocksize < nchar_adjusted:
                    fh.write(safename(taxon, mrbayes=mrbayes) + '\n')
                else:
                    fh.write(safename(taxon, mrbayes=mrbayes).ljust(namelength + 1))
                taxon_seq = cropped_matrix[taxon]
                for seek in range(0, nchar_adjusted, blocksize):
                    fh.write(taxon_seq[seek:seek + blocksize] + '\n')
                del taxon_seq
        fh.write(';\nend;\n')
        if append_sets:
            if codons_block:
                fh.write(self.append_sets(exclude=exclude, delete=delete, mrbayes=mrbayes, include_codons=False))
                fh.write(self.append_sets(exclude=exclude, delete=delete, mrbayes=mrbayes, codons_only=True))
            else:
                fh.write(self.append_sets(exclude=exclude, delete=delete, mrbayes=mrbayes))
    return filename