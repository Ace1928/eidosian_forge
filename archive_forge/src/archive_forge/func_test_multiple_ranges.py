import os
import shutil
import tempfile
import unittest
import patiencediff
from . import _patiencediff_py
def test_multiple_ranges(self):
    self.assertDiffBlocks('abcdefghijklmnop', 'abcXghiYZQRSTUVWXYZijklmnop', [(0, 0, 3), (6, 4, 3), (9, 20, 7)])
    self.assertDiffBlocks('ABCd efghIjk  L', 'AxyzBCn mo pqrstuvwI1 2  L', [(0, 0, 1), (1, 4, 2), (9, 19, 1), (12, 23, 3)])
    self.assertDiffBlocks('    trg nqqrq jura lbh nqq n svyr va gur qverpgbel.\n    """\n    gnxrf_netf = [\'svyr*\']\n    gnxrf_bcgvbaf = [\'ab-erphefr\']\n\n    qrs eha(frys, svyr_yvfg, ab_erphefr=Snyfr):\n        sebz omeyvo.nqq vzcbeg fzneg_nqq, nqq_ercbegre_cevag, nqq_ercbegre_ahyy\n        vs vf_dhvrg():\n            ercbegre = nqq_ercbegre_ahyy\n        ryfr:\n            ercbegre = nqq_ercbegre_cevag\n        fzneg_nqq(svyr_yvfg, abg ab_erphefr, ercbegre)\n\n\npynff pzq_zxqve(Pbzznaq):\n'.splitlines(True), '    trg nqqrq jura lbh nqq n svyr va gur qverpgbel.\n\n    --qel-eha jvyy fubj juvpu svyrf jbhyq or nqqrq, ohg abg npghnyyl\n    nqq gurz.\n    """\n    gnxrf_netf = [\'svyr*\']\n    gnxrf_bcgvbaf = [\'ab-erphefr\', \'qel-eha\']\n\n    qrs eha(frys, svyr_yvfg, ab_erphefr=Snyfr, qel_eha=Snyfr):\n        vzcbeg omeyvo.nqq\n\n        vs qel_eha:\n            vs vf_dhvrg():\n                # Guvf vf cbvagyrff, ohg V\'q engure abg envfr na reebe\n                npgvba = omeyvo.nqq.nqq_npgvba_ahyy\n            ryfr:\n  npgvba = omeyvo.nqq.nqq_npgvba_cevag\n        ryvs vf_dhvrg():\n            npgvba = omeyvo.nqq.nqq_npgvba_nqq\n        ryfr:\n       npgvba = omeyvo.nqq.nqq_npgvba_nqq_naq_cevag\n\n        omeyvo.nqq.fzneg_nqq(svyr_yvfg, abg ab_erphefr, npgvba)\n\n\npynff pzq_zxqve(Pbzznaq):\n'.splitlines(True), [(0, 0, 1), (1, 4, 2), (9, 19, 1), (12, 23, 3)])