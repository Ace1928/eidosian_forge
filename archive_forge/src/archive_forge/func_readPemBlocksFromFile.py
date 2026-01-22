import base64
import sys
def readPemBlocksFromFile(fileObj, *markers):
    startMarkers = dict(map(lambda x: (x[1], x[0]), enumerate(map(lambda y: y[0], markers))))
    stopMarkers = dict(map(lambda x: (x[1], x[0]), enumerate(map(lambda y: y[1], markers))))
    idx = -1
    substrate = ''
    certLines = []
    state = stSpam
    while True:
        certLine = fileObj.readline()
        if not certLine:
            break
        certLine = certLine.strip()
        if state == stSpam:
            if certLine in startMarkers:
                certLines = []
                idx = startMarkers[certLine]
                state = stHam
                continue
        if state == stHam:
            if certLine in stopMarkers and stopMarkers[certLine] == idx:
                state = stDump
            else:
                certLines.append(certLine)
        if state == stDump:
            if sys.version_info[0] <= 2:
                substrate = ''.join([base64.b64decode(x) for x in certLines])
            else:
                substrate = ''.encode().join([base64.b64decode(x.encode()) for x in certLines])
            break
    return (idx, substrate)