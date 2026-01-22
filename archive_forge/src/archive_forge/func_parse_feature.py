import warnings
import re
import sys
from collections import defaultdict
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonParserWarning
from typing import List
def parse_feature(self, feature_key, lines):
    """Parse a feature given as a list of strings into a tuple.

        Expects a feature as a list of strings, returns a tuple (key, location,
        qualifiers)

        For example given this GenBank feature::

             CDS             complement(join(490883..490885,1..879))
                             /locus_tag="NEQ001"
                             /note="conserved hypothetical [Methanococcus jannaschii];
                             COG1583:Uncharacterized ACR; IPR001472:Bipartite nuclear
                             localization signal; IPR002743: Protein of unknown
                             function DUF57"
                             /codon_start=1
                             /transl_table=11
                             /product="hypothetical protein"
                             /protein_id="NP_963295.1"
                             /db_xref="GI:41614797"
                             /db_xref="GeneID:2732620"
                             /translation="MRLLLELKALNSIDKKQLSNYLIQGFIYNILKNTEYSWLHNWKK
                             EKYFNFTLIPKKDIIENKRYYLIISSPDKRFIEVLHNKIKDLDIITIGLAQFQLRKTK
                             KFDPKLRFPWVTITPIVLREGKIVILKGDKYYKVFVKRLEELKKYNLIKKKEPILEEP
                             IEISLNQIKDGWKIIDVKDRYYDFRNKSFSAFSNWLRDLKEQSLRKYNNFCGKNFYFE
                             EAIFEGFTFYKTVSIRIRINRGEAVYIGTLWKELNVYRKLDKEEREFYKFLYDCGLGS
                             LNSMGFGFVNTKKNSAR"

        Then should give input key="CDS" and the rest of the data as a list of strings
        lines=["complement(join(490883..490885,1..879))", ..., "LNSMGFGFVNTKKNSAR"]
        where the leading spaces and trailing newlines have been removed.

        Returns tuple containing: (key as string, location string, qualifiers as list)
        as follows for this example:

        key = "CDS", string
        location = "complement(join(490883..490885,1..879))", string
        qualifiers = list of string tuples:

        [('locus_tag', '"NEQ001"'),
         ('note', '"conserved hypothetical [Methanococcus jannaschii];\\nCOG1583:..."'),
         ('codon_start', '1'),
         ('transl_table', '11'),
         ('product', '"hypothetical protein"'),
         ('protein_id', '"NP_963295.1"'),
         ('db_xref', '"GI:41614797"'),
         ('db_xref', '"GeneID:2732620"'),
         ('translation', '"MRLLLELKALNSIDKKQLSNYLIQGFIYNILKNTEYSWLHNWKK\\nEKYFNFT..."')]

        In the above example, the "note" and "translation" were edited for compactness,
        and they would contain multiple new line characters (displayed above as \\n)

        If a qualifier is quoted (in this case, everything except codon_start and
        transl_table) then the quotes are NOT removed.

        Note that no whitespace is removed.
        """
    iterator = (x for x in lines if x)
    try:
        line = next(iterator)
        feature_location = line.strip()
        while feature_location[-1:] == ',':
            line = next(iterator)
            feature_location += line.strip()
        if feature_location.count('(') > feature_location.count(')'):
            warnings.warn("Non-standard feature line wrapping (didn't break on comma)?", BiopythonParserWarning)
            while feature_location[-1:] == ',' or feature_location.count('(') > feature_location.count(')'):
                line = next(iterator)
                feature_location += line.strip()
        qualifiers = []
        for line_number, line in enumerate(iterator):
            if line_number == 0 and line.startswith(')'):
                feature_location += line.strip()
            elif line[0] == '/':
                i = line.find('=')
                key = line[1:i]
                value = line[i + 1:]
                if i and value.startswith(' ') and value.lstrip().startswith('"'):
                    warnings.warn('White space after equals in qualifier', BiopythonParserWarning)
                    value = value.lstrip()
                if i == -1:
                    key = line[1:]
                    qualifiers.append((key, None))
                elif not value:
                    qualifiers.append((key, ''))
                elif value == '"':
                    if self.debug:
                        print(f'Single quote {key}:{value}')
                    qualifiers.append((key, value))
                elif value[0] == '"':
                    value_list = [value]
                    while value_list[-1][-1] != '"':
                        value_list.append(next(iterator))
                    value = '\n'.join(value_list)
                    qualifiers.append((key, value))
                else:
                    qualifiers.append((key, value))
            else:
                assert len(qualifiers) > 0
                assert key == qualifiers[-1][0]
                if qualifiers[-1][1] is None:
                    raise StopIteration
                qualifiers[-1] = (key, qualifiers[-1][1] + '\n' + line)
        return (feature_key, feature_location, qualifiers)
    except StopIteration:
        raise ValueError("Problem with '%s' feature:\n%s" % (feature_key, '\n'.join(lines))) from None