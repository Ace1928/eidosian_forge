from __future__ import annotations
import inspect
import re
import types
import typing
from subprocess import PIPE, Popen
import gradio as gr
from app import demo as app
import os
def make_space(docs: dict, name: str, description: str, local_version: str | None, demo: str, space: str | None, repo: str | None, pypi_exists: bool, suppress_demo_check: bool=False):
    filtered_keys = [key for key in docs if key != '__meta__']
    if not suppress_demo_check and (demo.find("if __name__ == '__main__'") == -1 and demo.find('if __name__ == "__main__"') == -1):
        raise ValueError("The demo must be launched using `if __name__ == '__main__'`, otherwise the docs page will not function correctly.\n\nTo fix this error, launch the demo inside of an if statement like this:\n\nif __name__ == '__main__':\n    demo.launch()\n\nTo ignore this error pass `--suppress-demo-check` to the docs command.")
    demo = demo.replace('"""', '\\"\\"\\"')
    source = '\nimport gradio as gr\nfrom app import demo as app\nimport os\n'
    docs_classes = render_class_docs(filtered_keys, docs)
    source += f'\n_docs = {docs}\n\nabs_path = os.path.join(os.path.dirname(__file__), "css.css")\n\nwith gr.Blocks(\n    css=abs_path,\n    theme=gr.themes.Default(\n        font_mono=[\n            gr.themes.GoogleFont("Inconsolata"),\n            "monospace",\n        ],\n    ),\n) as demo:\n    gr.Markdown(\n"""\n# `{name}`\n\n<div style="display: flex; gap: 7px;">\n{render_version_badge(pypi_exists, local_version, name)} {render_github_badge(repo)} {render_discuss_badge(space)}\n</div>\n\n{description}\n""", elem_classes=["md-custom"], header_links=True)\n    app.render()\n    gr.Markdown(\n"""\n## Installation\n\n```bash\npip install {name}\n```\n\n## Usage\n\n```python\n{demo}\n```\n""", elem_classes=["md-custom"], header_links=True)\n\n{docs_classes}\n\n{render_additional_interfaces(docs['__meta__']['additional_interfaces'])}\n    demo.load(None, js=r"""{make_js(get_deep(docs, ['__meta__', 'additional_interfaces']), get_deep(docs, ['__meta__', 'user_fn_refs']))}\n""")\n\ndemo.launch()\n'
    return source