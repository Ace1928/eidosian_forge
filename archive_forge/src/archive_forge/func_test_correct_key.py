from matplotlib import pyplot as plt
import pytest
@pytest.mark.backend('gtk3agg', skip_on_importerror=True)
def test_correct_key():
    pytest.xfail('test_widget_send_event is not triggering key_press_event')
    from gi.repository import Gdk, Gtk
    fig = plt.figure()
    buf = []

    def send(event):
        for key, mod in [(Gdk.KEY_a, Gdk.ModifierType.SHIFT_MASK), (Gdk.KEY_a, 0), (Gdk.KEY_a, Gdk.ModifierType.CONTROL_MASK), (Gdk.KEY_agrave, 0), (Gdk.KEY_Control_L, Gdk.ModifierType.MOD1_MASK), (Gdk.KEY_Alt_L, Gdk.ModifierType.CONTROL_MASK), (Gdk.KEY_agrave, Gdk.ModifierType.CONTROL_MASK | Gdk.ModifierType.MOD1_MASK | Gdk.ModifierType.MOD4_MASK), (64790, 0), (Gdk.KEY_BackSpace, 0), (Gdk.KEY_BackSpace, Gdk.ModifierType.CONTROL_MASK)]:
            Gtk.test_widget_send_key(fig.canvas, key, mod)

    def receive(event):
        buf.append(event.key)
        if buf == ['A', 'a', 'ctrl+a', 'à', 'alt+control', 'ctrl+alt', 'ctrl+alt+super+à', 'backspace', 'ctrl+backspace']:
            plt.close(fig)
    fig.canvas.mpl_connect('draw_event', send)
    fig.canvas.mpl_connect('key_press_event', receive)
    plt.show()