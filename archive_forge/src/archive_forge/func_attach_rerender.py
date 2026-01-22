import inspect
import time
from typing import Iterable
from gradio_client.documentation import document_fn
import gradio as gr
import gradio as gr
def attach_rerender(evt_listener):
    return evt_listener(render_variables, [history, base_theme_dropdown] + theme_inputs, [history, secret_css, secret_font, output_code, current_theme], show_api=False).then(None, [secret_css, secret_font], None, js='(css, fonts) => {\n                    document.getElementById(\'theme_css\').innerHTML = css;\n                    let existing_font_links = document.querySelectorAll(\'link[rel="stylesheet"][href^="https://fonts.googleapis.com/css"]\');\n                    existing_font_links.forEach(link => {\n                        if (fonts.includes(link.href)) {\n                            fonts = fonts.filter(font => font != link.href);\n                        } else {\n                            link.remove();\n                        }\n                    });\n                    fonts.forEach(font => {\n                        let link = document.createElement(\'link\');\n                        link.rel = \'stylesheet\';\n                        link.href = font;\n                        document.head.appendChild(link);\n                    });\n                }', show_api=False)