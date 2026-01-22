import os
import pathlib
from gradio.themes.utils import ThemeAsset
def make_else_if(theme_asset):
    return f"\n        else if (theme == '{str(theme_asset[0].version)}') {{\n            var theme_css = `{theme_asset[1]._get_theme_css()}`\n        }}"