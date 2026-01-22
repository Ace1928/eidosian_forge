import abc
import hmac
import hashlib
import math
def expand(self, prk, info, outputSize):
    iterations = int(math.ceil(float(outputSize) / float(self.__class__.HASH_OUTPUT_SIZE)))
    mixin = bytearray()
    results = bytearray()
    remainingBytes = outputSize
    for i in range(self.getIterationStartOffset(), iterations + self.getIterationStartOffset()):
        mac = hmac.new(prk, digestmod=hashlib.sha256)
        mac.update(bytes(mixin))
        if info is not None:
            mac.update(bytes(info))
        updateChr = chr(i % 256)
        mac.update(updateChr.encode())
        stepResult = mac.digest()
        stepSize = min(remainingBytes, len(stepResult))
        results.extend(stepResult[:stepSize])
        mixin = stepResult
        remainingBytes -= stepSize
    return bytes(results)