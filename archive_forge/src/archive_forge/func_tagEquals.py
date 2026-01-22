import binascii
@staticmethod
def tagEquals(node, string):
    return node is not None and node.tag is not None and (node.tag == string)