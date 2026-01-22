import unittest
def test_walk_large_tree(self):
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.label import Label
    ' the tree\n        BoxLayout\n            BoxLayout\n            Label\n                10 labels\n            BoxLayout\n                10 labels\n            BoxLayout\n                Label\n            Label\n        '
    root = BoxLayout()
    tree = [root]
    box = BoxLayout()
    tree.append(box)
    root.add_widget(box)
    label = Label()
    tree.append(label)
    root.add_widget(label)
    for i in range(10):
        tree.append(Label())
        label.add_widget(tree[-1])
    box = BoxLayout()
    tree.append(box)
    root.add_widget(box)
    for i in range(10):
        tree.append(Label())
        box.add_widget(tree[-1])
    box = BoxLayout()
    tree.append(box)
    root.add_widget(box)
    tree.append(Label())
    box.add_widget(tree[-1])
    label = Label()
    tree.append(label)
    root.add_widget(label)

    def rotate(l, n):
        return l[n:] + l[:n]
    for i in range(len(tree)):
        rotated = rotate(tree, i)
        walked = [n for n in tree[i].walk(loopback=True)]
        walked_reversed = [n for n in tree[i].walk_reverse(loopback=True)]
        self.assertListEqual(rotated, walked)
        self.assertListEqual(walked, list(reversed(walked_reversed)))