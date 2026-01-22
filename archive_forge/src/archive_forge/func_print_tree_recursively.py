@staticmethod
def print_tree_recursively(node, depth):
    if not node:
        return
    if not node.isRed:
        depth += 1
    _IntervalTreeTester.print_tree_recursively(node.children[0], depth)
    align = 6 * depth
    if node.isRed:
        align += 3
    print(align * ' ', end=' ')
    if node.isRed:
        print('R', end=' ')
    else:
        print('B', end=' ')
    print(node.interval.lower())
    _IntervalTreeTester.print_tree_recursively(node.children[1], depth)