import ctypes
import weakref
import pyglet
from pyglet.gl import *
from pyglet.graphics import shader, vertexdomain
from pyglet.graphics.vertexarray import VertexArray
from pyglet.graphics.vertexbuffer import BufferObject
def draw_subset(self, vertex_lists):
    """Draw only some vertex lists in the batch.

        The use of this method is highly discouraged, as it is quite
        inefficient.  Usually an application can be redesigned so that batches
        can always be drawn in their entirety, using `draw`.

        The given vertex lists must belong to this batch; behaviour is
        undefined if this condition is not met.

        :Parameters:
            `vertex_lists` : sequence of `VertexList` or `IndexedVertexList`
                Vertex lists to draw.

        """

    def visit(group):
        group.set_state()
        domain_map = self.group_map[group]
        for (_, mode, _, _), domain in domain_map.items():
            for alist in vertex_lists:
                if alist.domain is domain:
                    alist.draw(mode)
        children = self.group_children.get(group)
        if children:
            children.sort()
            for child in children:
                if child.visible:
                    visit(child)
        group.unset_state()
    self.top_groups.sort()
    for group in self.top_groups:
        if group.visible:
            visit(group)