from nltk.tree.tree import Tree
def chomsky_normal_form(tree, factor='right', horzMarkov=None, vertMarkov=0, childChar='|', parentChar='^'):
    if horzMarkov is None:
        horzMarkov = 999
    nodeList = [(tree, [tree.label()])]
    while nodeList != []:
        node, parent = nodeList.pop()
        if isinstance(node, Tree):
            parentString = ''
            originalNode = node.label()
            if vertMarkov != 0 and node != tree and isinstance(node[0], Tree):
                parentString = '{}<{}>'.format(parentChar, '-'.join(parent))
                node.set_label(node.label() + parentString)
                parent = [originalNode] + parent[:vertMarkov - 1]
            for child in node:
                nodeList.append((child, parent))
            if len(node) > 2:
                childNodes = [child.label() for child in node]
                nodeCopy = node.copy()
                node[0:] = []
                curNode = node
                numChildren = len(nodeCopy)
                for i in range(1, numChildren - 1):
                    if factor == 'right':
                        newHead = '{}{}<{}>{}'.format(originalNode, childChar, '-'.join(childNodes[i:min([i + horzMarkov, numChildren])]), parentString)
                        newNode = Tree(newHead, [])
                        curNode[0:] = [nodeCopy.pop(0), newNode]
                    else:
                        newHead = '{}{}<{}>{}'.format(originalNode, childChar, '-'.join(childNodes[max([numChildren - i - horzMarkov, 0]):-i]), parentString)
                        newNode = Tree(newHead, [])
                        curNode[0:] = [newNode, nodeCopy.pop()]
                    curNode = newNode
                curNode[0:] = [child for child in nodeCopy]