import weakref
from enum import Enum
import threading
from typing import Tuple
import pyglet
from pyglet import gl
from pyglet.gl import gl_info
def delete_texture(self, texture_id):
    """Safely delete a Texture belonging to this context's object space.

        This method will delete the texture immediately via
        ``glDeleteTextures`` if the current context's object space is the same
        as this context's object space and it is called from the main thread.

        Otherwise, the texture will only be marked for deletion, postponing
        it until any context with the same object space becomes active again.

        This makes it safe to call from anywhere, including other threads.

        :Parameters:
            `texture_id` : int
                The OpenGL name of the Texture to delete.

        """
    if self._safe_to_operate_on_object_space():
        gl.glDeleteTextures(1, gl.GLuint(texture_id))
    else:
        self.object_space.doomed_textures.append(texture_id)