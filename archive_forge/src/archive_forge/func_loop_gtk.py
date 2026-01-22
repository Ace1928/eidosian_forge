import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
@register_integration('gtk')
def loop_gtk(kernel):
    """Start the kernel, coordinating with the GTK event loop"""
    from .gui.gtkembed import GTKEmbed
    gtk_kernel = GTKEmbed(kernel)
    gtk_kernel.start()
    kernel._gtk = gtk_kernel