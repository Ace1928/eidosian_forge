import inspect
import time
from typing import Iterable
from gradio_client.documentation import document_fn
import gradio as gr
import gradio as gr
def generate_theme_code(base_theme, final_theme, core_variables, final_main_fonts, final_mono_fonts):
    base_theme_name = base_theme
    base_theme = [theme for theme in themes if theme.__name__ == base_theme][0]()
    parameters = inspect.signature(base_theme.__init__).parameters
    primary_hue = parameters['primary_hue'].default
    secondary_hue = parameters['secondary_hue'].default
    neutral_hue = parameters['neutral_hue'].default
    text_size = parameters['text_size'].default
    spacing_size = parameters['spacing_size'].default
    radius_size = parameters['radius_size'].default
    font = parameters['font'].default
    font = [font] if not isinstance(font, Iterable) else font
    font = [gr.themes.Font(f) if not isinstance(f, gr.themes.Font) else f for f in font]
    font_mono = parameters['font_mono'].default
    font_mono = [font_mono] if not isinstance(font_mono, Iterable) else font_mono
    font_mono = [gr.themes.Font(f) if not isinstance(f, gr.themes.Font) else f for f in font_mono]
    core_diffs = {}
    specific_core_diffs = {}
    core_var_names = ['primary_hue', 'secondary_hue', 'neutral_hue', 'text_size', 'spacing_size', 'radius_size']
    for value_name, base_value, source_class, final_value in zip(core_var_names, [primary_hue, secondary_hue, neutral_hue, text_size, spacing_size, radius_size], [gr.themes.Color, gr.themes.Color, gr.themes.Color, gr.themes.Size, gr.themes.Size, gr.themes.Size], core_variables):
        if base_value.name != final_value:
            core_diffs[value_name] = final_value
        source_obj = [obj for obj in source_class.all if obj.name == final_value][0]
        final_attr_values = {}
        diff = False
        for attr in dir(source_obj):
            if attr in ['all', 'name', 'expand'] or attr.startswith('_'):
                continue
            final_theme_attr = value_name.split('_')[0] + '_' + (attr[1:] if source_class == gr.themes.Color else attr)
            final_attr_values[final_theme_attr] = getattr(final_theme, final_theme_attr)
            if getattr(source_obj, attr) != final_attr_values[final_theme_attr]:
                diff = True
        if diff:
            new_final_attr_values = {}
            for key, val in final_attr_values.items():
                if key.startswith(('primary_', 'secondary_', 'neutral_')):
                    color_key = 'c' + key.split('_')[-1]
                    new_final_attr_values[color_key] = val
                elif key.startswith(('text_', 'spacing_', 'radius_')):
                    size_key = key.split('_')[-1]
                    new_final_attr_values[size_key] = val
                else:
                    new_final_attr_values[key] = val
            specific_core_diffs[value_name] = (source_class, new_final_attr_values)
    font_diffs = {}
    final_main_fonts = [font for font in final_main_fonts if font[0]]
    final_mono_fonts = [font for font in final_mono_fonts if font[0]]
    font = font[:4]
    font_mono = font_mono[:4]
    for base_font_set, theme_font_set, font_set_name in [(font, final_main_fonts, 'font'), (font_mono, final_mono_fonts, 'font_mono')]:
        if len(base_font_set) != len(theme_font_set) or any((base_font.name != theme_font[0] or isinstance(base_font, gr.themes.GoogleFont) != theme_font[1] for base_font, theme_font in zip(base_font_set, theme_font_set))):
            font_diffs[font_set_name] = [f"gr.themes.GoogleFont('{font_name}')" if is_google_font else f"'{font_name}'" for font_name, is_google_font in theme_font_set]
    newline = '\n'
    core_diffs_code = ''
    if len(core_diffs) + len(specific_core_diffs) > 0:
        for var_name in core_var_names:
            if var_name in specific_core_diffs:
                cls, vals = specific_core_diffs[var_name]
                core_diffs_code += f'    {var_name}=gr.themes.{cls.__name__}({', '.join((f'{k}="{v}"' for k, v in vals.items()))}),\n'
            elif var_name in core_diffs:
                var_val = core_diffs[var_name]
                if var_name.endswith('_size'):
                    var_val = var_val.split('_')[-1]
                core_diffs_code += f'    {var_name}="{var_val}",\n'
    font_diffs_code = ''
    if len(font_diffs) > 0:
        font_diffs_code = ''.join([f'    {font_set_name}=[{', '.join(fonts)}],\n' for font_set_name, fonts in font_diffs.items()])
    var_diffs = {}
    for variable in flat_variables:
        base_theme_val = getattr(base_theme, variable)
        final_theme_val = getattr(final_theme, variable)
        if base_theme_val is None and variable.endswith('_dark'):
            base_theme_val = getattr(base_theme, variable[:-5])
        if base_theme_val != final_theme_val:
            var_diffs[variable] = getattr(final_theme, variable)
    newline = '\n'
    vars_diff_code = ''
    if len(var_diffs) > 0:
        vars_diff_code = f'.set(\n    {(',' + newline + '    ').join([f"{k}='{v}'" for k, v in var_diffs.items()])}\n)'
    output = f'\nimport gradio as gr\n\ntheme = gr.themes.{base_theme_name}({(newline if core_diffs_code or font_diffs_code else '')}{core_diffs_code}{font_diffs_code}){vars_diff_code}\n\nwith gr.Blocks(theme=theme) as demo:\n    ...'
    return output