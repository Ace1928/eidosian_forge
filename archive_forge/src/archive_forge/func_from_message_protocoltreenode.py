@staticmethod
def from_message_protocoltreenode(node):
    return MessageMetaAttributes(node['id'], node['from'], node['to'], node['notify'], node['t'], node['participant'], node['offline'], node['retry'])