def writeInternal(self, node, data):
    x = 1 + (0 if node.attributes is None else len(node.attributes) * 2) + (0 if not node.hasChildren() else 1) + (0 if node.data is None else 1)
    self.writeListStart(x, data)
    self.writeString(node.tag, data)
    self.writeAttributes(node.attributes, data)
    if node.data is not None:
        self.writeBytes(node.data, data)
    if node.hasChildren():
        self.writeListStart(len(node.children), data)
        for c in node.children:
            self.writeInternal(c, data)