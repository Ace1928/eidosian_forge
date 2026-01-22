import re
import itertools
@staticmethod
def createBytes(buffer, rsBlocks):
    offset = 0
    maxDcCount = 0
    maxEcCount = 0
    totalCodeCount = 0
    dcdata = []
    ecdata = []
    for block in rsBlocks:
        totalCodeCount += block.totalCount
        dcCount = block.dataCount
        ecCount = block.totalCount - dcCount
        maxDcCount = max(maxDcCount, dcCount)
        maxEcCount = max(maxEcCount, ecCount)
        dcdata.append(buffer.buffer[offset:offset + dcCount])
        offset += dcCount
        rsPoly = QRUtil.getErrorCorrectPolynomial(ecCount)
        rawPoly = QRPolynomial(dcdata[-1], rsPoly.getLength() - 1)
        modPoly = rawPoly.mod(rsPoly)
        rLen = rsPoly.getLength() - 1
        mLen = modPoly.getLength()
        ecdata.append([modPoly.get(i) if i >= 0 else 0 for i in range(mLen - rLen, mLen)])
    data = [d for dd in itertools.chain(zip_longest(*dcdata), zip_longest(*ecdata)) for d in dd if d is not None]
    return data