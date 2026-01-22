import logging
import warnings
from passlib import hash
from passlib.utils.compat import u
from passlib.tests.utils import TestCase, HandlerCase
from passlib.tests.test_handlers import UPASS_WAV
def test_96_full_verify(self):
    """test verify(full=True) flag"""

    def vpart(s, h):
        return self.handler.verify(s, h)

    def vfull(s, h):
        return self.handler.verify(s, h, full=True)
    h = '$scram$4096$QSXCR.Q6sek8bf92$sha-1=HZbuOlKbWl.eR8AfIposuKbhX30,sha-256=qXUXrlcvnaxxWG00DdRgVioR2gnUpuX5r.3EZ1rdhVY,sha-512=lzgniLFcvglRLS0gt.C4gy.NurS3OIOVRAU1zZOV4P.qFiVFO2/edGQSu/kD1LwdX0SNV/KsPdHSwEl5qRTuZQ'
    self.assertTrue(vfull('pencil', h))
    self.assertFalse(vfull('tape', h))
    h = '$scram$4096$QSXCR.Q6sek8bf92$sha-1=HZbuOlKbWl.eR8AfIposuKbhX30,sha-256=qXUXrlcvnaxxWG00DdRgVioR2gnUpuX5r.3EZ1rdhV,sha-512=lzgniLFcvglRLS0gt.C4gy.NurS3OIOVRAU1zZOV4P.qFiVFO2/edGQSu/kD1LwdX0SNV/KsPdHSwEl5qRTuZQ'
    self.assertRaises(ValueError, vfull, 'pencil', h)
    h = '$scram$4096$QSXCR.Q6sek8bf92$sha-1=HZbuOlKbWl.eR8AfIposuKbhX30,sha-256=qXUXrlcvnaxxWG00DdRgVioR2gnUpuX5r.3EZ1rdhVYa,sha-512=lzgniLFcvglRLS0gt.C4gy.NurS3OIOVRAU1zZOV4P.qFiVFO2/edGQSu/kD1LwdX0SNV/KsPdHSwEl5qRTuZQ'
    self.assertRaises(ValueError, vfull, 'pencil', h)
    h = '$scram$4096$QSXCR.Q6sek8bf92$sha-1=HZbuOlKbWl.eR8AfIposuKbhX30,sha-256=R7RJDWIbeKRTFwhE9oxh04kab0CllrQ3kCcpZUcligc,sha-512=lzgniLFcvglRLS0gt.C4gy.NurS3OIOVRAU1zZOV4P.qFiVFO2/edGQSu/kD1LwdX0SNV/KsPdHSwEl5qRTuZQ'
    self.assertTrue(vpart('tape', h))
    self.assertFalse(vpart('pencil', h))
    self.assertRaises(ValueError, vfull, 'pencil', h)
    self.assertRaises(ValueError, vfull, 'tape', h)