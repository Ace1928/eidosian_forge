import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
import numpy as np
import matplotlib.pyplot as plt
import io
def plot_fractal_to_texture(self, fractal_values):
    fig, ax = plt.subplots()
    cax = ax.imshow(fractal_values, extent=[-10, 10, 0, 10])
    fig.colorbar(cax)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    texture = Texture.create(size=fig.canvas.get_width_height(), colorfmt='rgba')
    texture.blit_buffer(buf.getvalue(), colorfmt='rgba', bufferfmt='ubyte')
    buf.close()
    plt.close(fig)
    self.img.texture = texture