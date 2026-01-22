import weakref
from enum import Enum
import threading
from typing import Tuple
import pyglet
from pyglet import gl
from pyglet.gl import gl_info
def delete_renderbuffer(self, rbo_id):
    """Safely delete a Renderbuffer Object belonging to this context's
        object space.

        This method behaves similarly to `delete_texture`, though for
        ``glDeleteRenderbuffers`` instead of ``glDeleteTextures``.

        :Parameters:
            `rbo_id` : int
                The OpenGL name of the Shader Program to delete.

        .. versionadded:: 2.0.10
        """
    if self._safe_to_operate_on_object_space():
        gl.glDeleteRenderbuffers(1, gl.GLuint(rbo_id))
    else:
        self.object_space.doomed_renderbuffers.append(rbo_id)