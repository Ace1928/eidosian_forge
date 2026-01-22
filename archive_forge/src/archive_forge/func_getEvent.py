import xml.sax
import xml.sax.handler
def getEvent(self):
    if not self.pulldom.firstEvent[1]:
        self.pulldom.lastEvent = self.pulldom.firstEvent
    while not self.pulldom.firstEvent[1]:
        buf = self.stream.read(self.bufsize)
        if not buf:
            self.parser.close()
            return None
        self.parser.feed(buf)
    rc = self.pulldom.firstEvent[1][0]
    self.pulldom.firstEvent[1] = self.pulldom.firstEvent[1][1]
    return rc