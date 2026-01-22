import os
from itertools import islice
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequentialAlignmentWriter
def get_spliced(self, starts, ends, strand=1):
    """Return a multiple alignment of the exact sequence range provided.

        Accepts two lists of start and end positions on target_seqname, representing
        exons to be spliced in silico.  Returns a *MultipleSeqAlignment* of the
        desired sequences spliced together.

        *starts* should be a list of 0-based start coordinates of segments in the reference.
        *ends* should be the list of the corresponding segment ends
        (in the half-open UCSC convention:
        http://genome.ucsc.edu/blog/the-ucsc-genome-browser-coordinate-counting-systems/).

        To ask for the alignment portion corresponding to the first 100
        nucleotides of the reference sequence, you would use
        ``search([0], [100])``
        """
    if strand not in (1, -1):
        raise ValueError(f'Strand must be 1 or -1, got {strand}')
    fetched = list(self.search(starts, ends))
    expected_letters = sum((end - start for start, end in zip(starts, ends)))
    if len(fetched) == 0:
        return MultipleSeqAlignment([SeqRecord(Seq('N' * expected_letters), id=self._target_seqname)])
    all_seqnames = {sequence.id for multiseq in fetched for sequence in multiseq}
    split_by_position = {seq_name: {} for seq_name in all_seqnames}
    total_rec_length = 0
    ref_first_strand = None
    for multiseq in fetched:
        for seqrec in multiseq:
            if seqrec.id == self._target_seqname:
                try:
                    if ref_first_strand is None:
                        ref_first_strand = seqrec.annotations['strand']
                        if ref_first_strand not in (1, -1):
                            raise ValueError('Strand must be 1 or -1')
                    elif ref_first_strand != seqrec.annotations['strand']:
                        raise ValueError("Encountered strand='%s' on target seqname, expected '%s'" % (seqrec.annotations['strand'], ref_first_strand))
                except KeyError:
                    raise ValueError('No strand information for target seqname (%s)' % self._target_seqname) from None
                rec_length = len(seqrec)
                rec_start = seqrec.annotations['start']
                ungapped_length = seqrec.annotations['size']
                rec_end = rec_start + ungapped_length - 1
                total_rec_length += ungapped_length
                for seqrec in multiseq:
                    for pos in range(rec_start, rec_end + 1):
                        split_by_position[seqrec.id][pos] = ''
                break
        else:
            raise ValueError(f'Did not find {self._target_seqname} in alignment bundle')
        real_pos = rec_start
        for gapped_pos in range(rec_length):
            for seqrec in multiseq:
                if seqrec.id == self._target_seqname:
                    track_val = seqrec.seq[gapped_pos]
                split_by_position[seqrec.id][real_pos] += seqrec.seq[gapped_pos]
            if track_val != '-' and real_pos < rec_end:
                real_pos += 1
    if len(split_by_position[self._target_seqname]) != total_rec_length:
        raise ValueError('Target seqname (%s) has %s records, expected %s' % (self._target_seqname, len(split_by_position[self._target_seqname]), total_rec_length))
    realpos_to_len = {pos: len(gapped_fragment) for pos, gapped_fragment in split_by_position[self._target_seqname].items() if len(gapped_fragment) > 1}
    subseq = {}
    for seqid in all_seqnames:
        seq_split = split_by_position[seqid]
        seq_splice = []
        filler_char = 'N' if seqid == self._target_seqname else '-'
        append = seq_splice.append
        for exonstart, exonend in zip(starts, ends):
            for real_pos in range(exonstart, exonend):
                if real_pos in seq_split:
                    append(seq_split[real_pos])
                elif real_pos in realpos_to_len:
                    append(filler_char * realpos_to_len[real_pos])
                else:
                    append(filler_char)
        subseq[seqid] = ''.join(seq_splice)
    if len(subseq[self._target_seqname].replace('-', '')) != expected_letters:
        raise ValueError('Returning %s letters for target seqname (%s), expected %s' % (len(subseq[self._target_seqname].replace('-', '')), self._target_seqname, expected_letters))
    ref_subseq_len = len(subseq[self._target_seqname])
    for seqid, seq in subseq.items():
        if len(seq) != ref_subseq_len:
            raise ValueError('Returning length %s for %s, expected %s' % (len(seq), seqid, ref_subseq_len))
    result_multiseq = []
    for seqid, seq in subseq.items():
        seq = Seq(seq)
        seq = seq if strand == ref_first_strand else seq.reverse_complement()
        result_multiseq.append(SeqRecord(seq, id=seqid, name=seqid, description=''))
    return MultipleSeqAlignment(result_multiseq)