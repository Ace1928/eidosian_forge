def writeBytes(self, bytes_, data, packed=False):
    bytes__ = []
    for b in bytes_:
        if type(b) is int:
            bytes__.append(b)
        else:
            bytes__.append(ord(b))
    size = len(bytes__)
    toWrite = bytes__
    if size >= 1048576:
        data.append(254)
        self.writeInt31(size, data)
    elif size >= 256:
        data.append(253)
        self.writeInt20(size, data)
    else:
        r = None
        if packed:
            if size < 128:
                r = self.tryPackAndWriteHeader(255, bytes__, data)
                if r is None:
                    r = self.tryPackAndWriteHeader(251, bytes__, data)
        if r is None:
            data.append(252)
            self.writeInt8(size, data)
        else:
            toWrite = r
    data.extend(toWrite)