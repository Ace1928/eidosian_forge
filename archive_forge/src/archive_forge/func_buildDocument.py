import xml.sax
import xml.sax.handler
def buildDocument(self, uri, tagname):
    node = self.documentFactory.createDocument(uri, tagname, None)
    self.document = node
    self.lastEvent[1] = [(START_DOCUMENT, node), None]
    self.lastEvent = self.lastEvent[1]
    self.push(node)
    for e in self.pending_events:
        if e[0][0] == PROCESSING_INSTRUCTION:
            _, target, data = e[0]
            n = self.document.createProcessingInstruction(target, data)
            e[0] = (PROCESSING_INSTRUCTION, n)
        elif e[0][0] == COMMENT:
            n = self.document.createComment(e[0][1])
            e[0] = (COMMENT, n)
        else:
            raise AssertionError('Unknown pending event ', e[0][0])
        self.lastEvent[1] = e
        self.lastEvent = e
    self.pending_events = None
    return node.firstChild