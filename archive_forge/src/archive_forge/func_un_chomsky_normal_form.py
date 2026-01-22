from nltk.tree.tree import Tree
def un_chomsky_normal_form(tree, expandUnary=True, childChar='|', parentChar='^', unaryChar='+'):
    nodeList = [(tree, [])]
    while nodeList != []:
        node, parent = nodeList.pop()
        if isinstance(node, Tree):
            childIndex = node.label().find(childChar)
            if childIndex != -1:
                nodeIndex = parent.index(node)
                parent.remove(parent[nodeIndex])
                if nodeIndex == 0:
                    parent.insert(0, node[0])
                    parent.insert(1, node[1])
                else:
                    parent.extend([node[0], node[1]])
                node = parent
            else:
                parentIndex = node.label().find(parentChar)
                if parentIndex != -1:
                    node.set_label(node.label()[:parentIndex])
                if expandUnary == True:
                    unaryIndex = node.label().find(unaryChar)
                    if unaryIndex != -1:
                        newNode = Tree(node.label()[unaryIndex + 1:], [i for i in node])
                        node.set_label(node.label()[:unaryIndex])
                        node[0:] = [newNode]
            for child in node:
                nodeList.append((child, node))