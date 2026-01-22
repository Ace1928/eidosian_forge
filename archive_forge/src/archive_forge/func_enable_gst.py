import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
def enable_gst():
    if _check_enabled('gst'):
        return
    gi.require_version('Gst', '0.10')
    from gi.repository import Gst
    _patch_module('gst', Gst)
    _install_enums(Gst)
    _patch(Gst, 'registry_get_default', Gst.Registry.get_default)
    _patch(Gst, 'element_register', Gst.Element.register)
    _patch(Gst, 'element_factory_make', Gst.ElementFactory.make)
    _patch(Gst, 'caps_new_any', Gst.Caps.new_any)
    _patch(Gst, 'get_pygst_version', lambda: (0, 10, 19))
    _patch(Gst, 'get_gst_version', lambda: (0, 10, 40))
    from gi.repository import GstInterfaces
    _patch_module('gst.interfaces', GstInterfaces)
    _install_enums(GstInterfaces)
    from gi.repository import GstAudio
    _patch_module('gst.audio', GstAudio)
    _install_enums(GstAudio)
    from gi.repository import GstVideo
    _patch_module('gst.video', GstVideo)
    _install_enums(GstVideo)
    from gi.repository import GstBase
    _patch_module('gst.base', GstBase)
    _install_enums(GstBase)
    _patch(Gst, 'BaseTransform', GstBase.BaseTransform)
    _patch(Gst, 'BaseSink', GstBase.BaseSink)
    from gi.repository import GstController
    _patch_module('gst.controller', GstController)
    _install_enums(GstController, dest=Gst)
    from gi.repository import GstPbutils
    _patch_module('gst.pbutils', GstPbutils)
    _install_enums(GstPbutils)