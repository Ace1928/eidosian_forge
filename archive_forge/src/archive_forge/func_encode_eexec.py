import fontTools
from fontTools.misc import eexec
from fontTools.misc.macCreatorType import getMacCreatorAndType
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes
from fontTools.misc.psOperators import (
from fontTools.encodings.StandardEncoding import StandardEncoding
import os
import re
def encode_eexec(self, eexec_dict):
    lines = []
    RD_key, ND_key, NP_key = (None, None, None)
    lenIV = 4
    subrs = std_subrs
    sortedItems = sorted(eexec_dict.items(), key=lambda item: item[0] != 'Private')
    for key, value in sortedItems:
        if key == 'Private':
            pr = eexec_dict['Private']
            size = 3
            for subkey in Private_dictionary_keys:
                size += int(subkey in pr)
            lines.append(b'dup /Private')
            lines.append(self._tobytes(f'{size} dict dup begin'))
            for subkey, subvalue in pr.items():
                if not RD_key and subvalue == RD_value:
                    RD_key = subkey
                elif not ND_key and subvalue in ND_values:
                    ND_key = subkey
                elif not NP_key and subvalue in PD_values:
                    NP_key = subkey
                if subkey == 'lenIV':
                    lenIV = subvalue
                if subkey == 'OtherSubrs':
                    lines.append(self._tobytes(hintothers))
                elif subkey == 'Subrs':
                    for subr_bin in subvalue:
                        subr_bin.compile()
                    subrs = [subr_bin.bytecode for subr_bin in subvalue]
                    lines.append(f'/Subrs {len(subrs)} array'.encode('ascii'))
                    for i, subr_bin in enumerate(subrs):
                        encrypted_subr, R = eexec.encrypt(bytesjoin([char_IV[:lenIV], subr_bin]), 4330)
                        lines.append(bytesjoin([self._tobytes(f'dup {i} {len(encrypted_subr)} {RD_key} '), encrypted_subr, self._tobytes(f' {NP_key}')]))
                    lines.append(b'def')
                    lines.append(b'put')
                else:
                    lines.extend(self._make_lines(subkey, subvalue))
        elif key == 'CharStrings':
            lines.append(b'dup /CharStrings')
            lines.append(self._tobytes(f'{len(eexec_dict['CharStrings'])} dict dup begin'))
            for glyph_name, char_bin in eexec_dict['CharStrings'].items():
                char_bin.compile()
                encrypted_char, R = eexec.encrypt(bytesjoin([char_IV[:lenIV], char_bin.bytecode]), 4330)
                lines.append(bytesjoin([self._tobytes(f'/{glyph_name} {len(encrypted_char)} {RD_key} '), encrypted_char, self._tobytes(f' {ND_key}')]))
            lines.append(b'end put')
        else:
            lines.extend(self._make_lines(key, value))
    lines.extend([b'end', b'dup /FontName get exch definefont pop', b'mark', b'currentfile closefile\n'])
    eexec_portion = bytesjoin(lines, '\n')
    encrypted_eexec, R = eexec.encrypt(bytesjoin([eexec_IV, eexec_portion]), 55665)
    return encrypted_eexec