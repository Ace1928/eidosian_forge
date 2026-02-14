from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union
from eidos_utility import get_universal_logger


T = TypeVar("T")


class Node(Generic[T]):
    """Represents a node in the Red-Black Tree."""

    def __init__(
        self,
        key: Optional[T] = None,
        color: str = "RED",
        left: Optional["Node[T]"] = None,
        right: Optional["Node[T]"] = None,
        parent: Optional["Node[T]"] = None,
    ):
        self.key = key
        self.color = color
        self.left = left
        self.right = right
        self.parent = parent

    def __str__(self):
        return f"Node(key={self.key}, color={self.color})"

    def __repr__(self):
        return f"Node(key={self.key}, color={self.color})"


class RedBlackTree(Generic[T]):
    """Implements a Red-Black Tree data structure."""

    def __init__(self):
        self.logger = get_universal_logger(name="red_black_tree_logger")

        self.NIL: Node[T] = Node()  # Sentinel node
        self.NIL.color = "BLACK"
        self.root = self.NIL
        self.logger.debug("RedBlackTree initialized")

    @property
    def _left_rotate_profiled(self):
        return self.profiler.profile(self._left_rotate)

    def _left_rotate(self, x: Node[T]) -> None:
        """Performs a left rotation on the tree."""
        self.logger.debug(f"Left rotate on node {x}")
        y = x.right
        if y != self.NIL:
            x.right = y.left
            if y.left != self.NIL:
                y.left.parent = x
            y.parent = x.parent
            if x.parent == self.NIL:
                self.root = y
            elif x.parent is not None and x.parent.left == x:
                x.parent.left = y
            elif x.parent is not None:
                x.parent.right = y
            y.left = x
            x.parent = y
        else:
            self.logger.warning(
                f"Attempted left rotate on node {x} with NIL right child."
            )
        self.logger.debug(f"Left rotate complete")

    @property
    def _right_rotate_profiled(self):
        return self.profiler.profile(self._right_rotate)

    def _right_rotate(self, y: Node[T]) -> None:
        """Performs a right rotation on the tree."""
        self.logger.debug(f"Right rotate on node {y}")
        x = y.left
        if x != self.NIL:
            y.left = x.right
            if x.right != self.NIL:
                x.right.parent = y
            x.parent = y.parent
            if y.parent == self.NIL:
                self.root = x
            elif y.parent is not None and y.parent.right == y:
                y.parent.right = x
            elif y.parent is not None:
                y.parent.left = x
            x.right = y
            y.parent = x
        else:
            self.logger.warning(
                f"Attempted right rotate on node {y} with NIL left child."
            )
        self.logger.debug(f"Right rotate complete")

    @property
    def insert_profiled(self):
        return self.profiler.profile(self.insert)

    def insert(self, key: T) -> None:
        """Inserts a new key into the tree."""
        self.logger.debug(f"Inserting key {key}")
        z = Node(key=key, color="RED", left=self.NIL, right=self.NIL)
        y = self.NIL
        x = self.root
        while x != self.NIL:
            y = x
            if z.key is not None and x.key is not None and z.key < x.key:
                x = x.left
            else:
                x = x.right
        z.parent = y
        if y == self.NIL:
            self.root = z
        elif y.key is not None and z.key is not None and z.key < y.key:
            y.left = z
        else:
            y.right = z
        self._insert_fixup(z)
        self.logger.debug(f"Insertion of key {key} complete")

    @property
    def _insert_fixup_profiled(self):
        return self.profiler.profile(self._insert_fixup)

    def _insert_fixup(self, z: Node[T]) -> None:
        """Restores the Red-Black Tree properties after insertion."""
        self.logger.debug(f"Insert fixup on node {z}")
        while z.parent is not None and z.parent.color == "RED":
            if z.parent.parent is not None and z.parent == z.parent.parent.left:
                y = z.parent.parent.right
                if y is not None and y.color == "RED":
                    z.parent.color = "BLACK"
                    y.color = "BLACK"
                    z.parent.parent.color = "RED"
                    z = z.parent.parent
                else:
                    if z == z.parent.right:
                        z = z.parent
                        self._left_rotate_profiled(z)
                    if z.parent.parent is not None:
                        z.parent.color = "BLACK"
                        z.parent.parent.color = "RED"
                        self._right_rotate_profiled(z.parent.parent)
            elif z.parent.parent is not None:
                y = z.parent.parent.left
                if y is not None and y.color == "RED":
                    z.parent.color = "BLACK"
                    y.color = "BLACK"
                    z.parent.parent.color = "RED"
                    z = z.parent.parent
                else:
                    if z == z.parent.left:
                        z = z.parent
                        self._right_rotate_profiled(z)
                    if z.parent.parent is not None:
                        z.parent.color = "BLACK"
                        z.parent.parent.color = "RED"
                        self._left_rotate_profiled(z.parent.parent)
            if z == self.root:
                break
        self.root.color = "BLACK"
        self.logger.debug(f"Insert fixup complete")

    @property
    def _transplant_profiled(self):
        return self.profiler.profile(self._transplant)

    def _transplant(self, u: Node[T], v: Node[T]) -> None:
        """Replaces subtree rooted at node u with subtree rooted at node v."""
        self.logger.debug(f"Transplant node {u} with {v}", obj={"u": u, "v": v})
        if u.parent == self.NIL:
            self.root = v
        elif u.parent is not None and u.parent.left == u:
            u.parent.left = v
        elif u.parent is not None:
            u.parent.right = v
        if v != self.NIL:
            v.parent = u.parent
        self.logger.debug(f"Transplant complete")

    @property
    def delete_profiled(self):
        return self.profiler.profile(self.delete)

    def delete(self, key: T) -> None:
        """Deletes a node with the given key from the tree."""
        self.logger.debug(f"Deleting key {key}")
        z = self.search(key)
        if z == self.NIL:
            self.logger.debug(f"Key {key} not found for deletion")
            return  # Key not found
        y = z
        y_original_color = y.color
        if z.left == self.NIL:
            x = z.right
            self._transplant_profiled(z, z.right)
        elif z.right == self.NIL:
            x = z.left
            self._transplant_profiled(z, z.left)
        else:
            y = self._minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent == z:
                if x != self.NIL:
                    x.parent = y
            else:
                self._transplant_profiled(y, y.right)
                y.right = z.right
                if y.right != self.NIL:
                    y.right.parent = y
            self._transplant_profiled(z, y)
            y.left = z.left
            if y.left != self.NIL:
                y.left.parent = y
            y.color = z.color
        if y_original_color == "BLACK":
            self._delete_fixup(x)
        self.logger.debug(f"Deletion of key {key} complete")

    @property
    def _delete_fixup_profiled(self):
        return self.profiler.profile(self._delete_fixup)

    def _delete_fixup(self, x: Node[T]) -> None:
        """Restores the Red-Black Tree properties after deletion."""
        self.logger.debug(f"Delete fixup on node {x}")
        while x != self.root and x.color == "BLACK":
            if x.parent is not None and x == x.parent.left:
                w = x.parent.right
                if w is not None and w.color == "RED":
                    w.color = "BLACK"
                    x.parent.color = "RED"
                    self._left_rotate_profiled(x.parent)
                    w = x.parent.right
                if (
                    w is not None
                    and w.left != self.NIL
                    and w.right != self.NIL
                    and w.left.color == "BLACK"
                    and w.right.color == "BLACK"
                ):
                    w.color = "RED"
                    x = x.parent
                else:
                    if (
                        w is not None
                        and w.right != self.NIL
                        and w.right.color == "BLACK"
                    ):
                        w.left.color = "BLACK"
                        w.color = "RED"
                        self._right_rotate_profiled(w)
                        w = x.parent.right
                    if w is not None:
                        w.color = x.parent.color if x.parent is not None else "BLACK"
                        if x.parent is not None:
                            x.parent.color = "BLACK"
                        if w.right != self.NIL:
                            w.right.color = "BLACK"
                        self._left_rotate_profiled(x.parent)
                    x = self.root
            elif x.parent is not None:
                w = x.parent.left
                if w is not None and w.color == "RED":
                    w.color = "BLACK"
                    x.parent.color = "RED"
                    self._right_rotate_profiled(x.parent)
                    w = x.parent.left
                if (
                    w is not None
                    and w.right != self.NIL
                    and w.left != self.NIL
                    and w.right.color == "BLACK"
                    and w.left.color == "BLACK"
                ):
                    w.color = "RED"
                    x = x.parent
                else:
                    if w is not None and w.left != self.NIL and w.left.color == "BLACK":
                        w.right.color = "BLACK"
                        w.color = "RED"
                        self._left_rotate_profiled(w)
                        w = x.parent.left
                    if w is not None:
                        w.color = x.parent.color if x.parent is not None else "BLACK"
                        if x.parent is not None:
                            x.parent.color = "BLACK"
                        if w.left != self.NIL:
                            w.left.color = "BLACK"
                        self._right_rotate_profiled(x.parent)
                    x = self.root
        if x is not None:
            x.color = "BLACK"
        self.logger.debug(f"Delete fixup complete")

    @property
    def search_profiled(self):
        return self.profiler.profile(self.search)

    def search(self, key: T) -> Node[T]:
        """Searches for a node with the given key."""
        self.logger.debug(f"Searching for key {key}")
        x = self.root
        while x != self.NIL and key != x.key:
            if x.key is not None and key < x.key:
                x = x.left
            else:
                x = x.right
        self.logger.debug(
            f"Search for key {key} complete, found {x}", obj={"key": key, "found": x}
        )
        return x

    @property
    def _minimum_profiled(self):
        return self.profiler.profile(self._minimum)

    def _minimum(self, x: Node[T]) -> Node[T]:
        """Returns the node with the minimum key in the subtree rooted at x."""
        self.logger.debug(f"Finding minimum in subtree rooted at {x}")
        while x.left != self.NIL:
            x = x.left
        self.logger.debug(f"Minimum found: {x}")
        return x

    @property
    def _maximum_profiled(self):
        return self.profiler.profile(self._maximum)

    def _maximum(self, x: Node[T]) -> Node[T]:
        """Returns the node with the maximum key in the subtree rooted at x."""
        self.logger.debug(f"Finding maximum in subtree rooted at {x}")
        while x.right != self.NIL:
            x = x.right
        self.logger.debug(f"Maximum found: {x}")
        return x

    def inorder_walk(
        self, node: Optional[Node[T]] = None, result: Optional[List[T]] = None
    ) -> List[T]:
        """Performs an inorder traversal of the tree."""
        if node is None:
            node = self.root
        if result is None:
            result = []
        if node != self.NIL:
            self.inorder_walk(node.left, result)
            if node.key is not None:
                result.append(node.key)
            self.inorder_walk(node.right, result)
        return result

    def preorder_walk(
        self, node: Optional[Node[T]] = None, result: Optional[List[T]] = None
    ) -> List[T]:
        """Performs a preorder traversal of the tree."""
        if node is None:
            node = self.root
        if result is None:
            result = []
        if node != self.NIL:
            if node.key is not None:
                result.append(node.key)
            self.preorder_walk(node.left, result)
            self.preorder_walk(node.right, result)
        return result

    def postorder_walk(
        self, node: Optional[Node[T]] = None, result: Optional[List[T]] = None
    ) -> List[T]:
        """Performs a postorder traversal of the tree."""
        if node is None:
            node = self.root
        if result is None:
            result = []
        if node != self.NIL:
            self.postorder_walk(node.left, result)
            self.postorder_walk(node.right, result)
            if node.key is not None:
                result.append(node.key)
        return result

    def get_height(self, node: Optional[Node[T]] = None) -> int:
        """Calculates the height of the tree."""
        if node is None:
            node = self.root
        if node == self.NIL:
            return 0
        else:
            return 1 + max(self.get_height(node.left), self.get_height(node.right))

    def is_empty(self) -> bool:
        """Checks if the tree is empty."""
        return self.root == self.NIL

    def clear(self) -> None:
        """Clears the tree."""
        self.root = self.NIL
        self.logger.debug("Tree cleared")

    def __str__(self) -> str:
        """Returns a string representation of the tree."""
        return f"RedBlackTree(root={self.root}, height={self.get_height()})"

    def __repr__(self) -> str:
        return f"RedBlackTree(root={self.root}, height={self.get_height()})"


if __name__ == "__main__":
    logger = UniversalLogger(name="main_red_black_tree_logger").get_logger()
    tree = RedBlackTree[int](logger=logger)
    keys = [7, 3, 18, 10, 22, 8, 11, 26, 2, 6, 13]
    for key in keys:
        tree.insert_profiled(key)
    logger.debug("Tree after insertions")
    print("Inorder Traversal:", tree.inorder_walk())
    print("Preorder Traversal:", tree.preorder_walk())
    print("Postorder Traversal:", tree.postorder_walk())
    print("Tree Height:", tree.get_height())
    print("Tree:", tree)
    tree.delete_profiled(10)
    logger.debug("Tree after deletion of 10")
    print("Inorder Traversal after deletion:", tree.inorder_walk())
    print("Tree Height after deletion:", tree.get_height())
    print("Tree after deletion:", tree)
    tree.clear()
    logger.debug("Tree after clear")
    print("Is tree empty?", tree.is_empty())
    print("Tree after clear:", tree)
