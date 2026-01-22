import re
import itertools
@classmethod
def getLostPoint(cls, qrCode):
    lostPoint = 0
    lostPoint += cls.maskScoreRule1vert(qrCode.modules)
    lostPoint += cls.maskScoreRule1vert(zip(*qrCode.modules))
    lostPoint += cls.maskScoreRule2(qrCode.modules)
    lostPoint += cls.maskScoreRule3hor(qrCode.modules)
    lostPoint += cls.maskScoreRule3hor(zip(*qrCode.modules))
    lostPoint += cls.maskScoreRule4(qrCode.modules)
    return lostPoint