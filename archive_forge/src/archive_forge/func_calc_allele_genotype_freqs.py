import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def calc_allele_genotype_freqs(self, fname):
    """Calculate allele and genotype frequencies per locus and per sample.

        Parameters:
         - fname - file name

        Returns tuple with 2 elements:
         - Population iterator with

           - population name
           - Locus dictionary with key = locus name and content tuple as
             Genotype List with
             (Allele1, Allele2, observed, expected)
             (expected homozygotes, observed hm,
             expected heterozygotes, observed ht)
             Allele frequency/Fis dictionary with allele as key and
             (count, frequency, Fis Weir & Cockerham)
           - Totals as a pair
           - count
           - Fis Weir & Cockerham,
           - Fis Robertson & Hill

         - Locus iterator with

           - Locus name
           - allele list
           - Population list with a triple

             - population name
             - list of allele frequencies in the same order as allele list above
             - number of genes

        Will create a file called fname.INF

        """
    self._run_genepop(['INF'], [5, 1], fname)

    def pop_parser(self):
        if hasattr(self, 'old_line'):
            line = self.old_line
            del self.old_line
        else:
            line = self.stream.readline()
        loci_content = {}
        while line != '':
            line = line.rstrip()
            if 'Tables of allelic frequencies for each locus' in line:
                return (self.curr_pop, loci_content)
            match = re.match('.*Pop: (.+) Locus: (.+)', line)
            if match is not None:
                pop = match.group(1).rstrip()
                locus = match.group(2)
                if not hasattr(self, 'first_locus'):
                    self.first_locus = locus
                if hasattr(self, 'curr_pop'):
                    if self.first_locus == locus:
                        old_pop = self.curr_pop
                        self.old_line = line
                        del self.first_locus
                        del self.curr_pop
                        return (old_pop, loci_content)
                self.curr_pop = pop
            else:
                line = self.stream.readline()
                continue
            geno_list = []
            line = self.stream.readline()
            if 'No data' in line:
                continue
            while 'Genotypes  Obs.' not in line:
                line = self.stream.readline()
            while line != '\n':
                m2 = re.match(' +([0-9]+) , ([0-9]+) *([0-9]+) *(.+)', line)
                if m2 is not None:
                    geno_list.append((_gp_int(m2.group(1)), _gp_int(m2.group(2)), _gp_int(m2.group(3)), _gp_float(m2.group(4))))
                else:
                    line = self.stream.readline()
                    continue
                line = self.stream.readline()
            while 'Expected number of ho' not in line:
                line = self.stream.readline()
            expHo = _gp_float(line[38:])
            line = self.stream.readline()
            obsHo = _gp_int(line[38:])
            line = self.stream.readline()
            expHe = _gp_float(line[38:])
            line = self.stream.readline()
            obsHe = _gp_int(line[38:])
            line = self.stream.readline()
            while 'Sample count' not in line:
                line = self.stream.readline()
            line = self.stream.readline()
            freq_fis = {}
            overall_fis = None
            while '----' not in line:
                vals = [x for x in line.rstrip().split(' ') if x != '']
                if vals[0] == 'Tot':
                    overall_fis = (_gp_int(vals[1]), _gp_float(vals[2]), _gp_float(vals[3]))
                else:
                    freq_fis[_gp_int(vals[0])] = (_gp_int(vals[1]), _gp_float(vals[2]), _gp_float(vals[3]))
                line = self.stream.readline()
            loci_content[locus] = (geno_list, (expHo, obsHo, expHe, obsHe), freq_fis, overall_fis)
        self.done = True
        raise StopIteration

    def locus_parser(self):
        line = self.stream.readline()
        while line != '':
            line = line.rstrip()
            match = re.match(' Locus: (.+)', line)
            if match is not None:
                locus = match.group(1)
                alleles, table = _read_allele_freq_table(self.stream)
                return (locus, alleles, table)
            line = self.stream.readline()
        self.done = True
        raise StopIteration
    shutil.copyfile(fname + '.INF', fname + '.IN2')
    pop_iter = _FileIterator(pop_parser, fname + '.INF')
    locus_iter = _FileIterator(locus_parser, fname + '.IN2')
    return (pop_iter, locus_iter)