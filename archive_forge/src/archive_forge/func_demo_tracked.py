import gradio as gr
from gradio.routes import App
from gradio.utils import BaseReloader
def demo_tracked(self) -> bool:
    return self.current_cell in self._running