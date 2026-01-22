import pyglet
from pyglet import gl
from pyglet import graphics
from pyglet.gl import current_context
from pyglet.math import Mat4, Vec3
from pyglet.graphics import shader
from .codecs import registry as _codec_registry
from .codecs import add_default_codecs as _add_default_codecs
def get_default_textured_shader():
    return pyglet.gl.current_context.create_program((TexturedMaterialGroup.default_vert_src, 'vertex'), (TexturedMaterialGroup.default_frag_src, 'fragment'))