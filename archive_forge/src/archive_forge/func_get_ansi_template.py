from warnings import warn
def get_ansi_template():
    try:
        from jinja2 import Template
    except ImportError:
        raise ImportError("please install the 'jinja2' package")
    return Template('\n    {%- for func_key in func_data.keys() -%}\n        Function name: \x1b[34m{{func_data[func_key][\'funcname\']}}\x1b[39;49;00m\n        {%- if func_data[func_key][\'filename\'] -%}\n        {{\'\n\'}}In file: \x1b[34m{{func_data[func_key][\'filename\'] -}}\x1b[39;49;00m\n        {%- endif -%}\n        {{\'\n\'}}With signature: \x1b[34m{{func_key[1]}}\x1b[39;49;00m\n        {{- "\n" -}}\n        {%- for num, line, hl, hc in func_data[func_key][\'pygments_lines\'] -%}\n                {{-\'\n\'}}{{ num}}: {{hc-}}\n                {%- if func_data[func_key][\'ir_lines\'][num] -%}\n                    {%- for ir_line, ir_line_type in func_data[func_key][\'ir_lines\'][num] %}\n                        {{-\'\n\'}}--{{- \' \'*func_data[func_key][\'python_indent\'][num]}}\n                        {{- \' \'*(func_data[func_key][\'ir_indent\'][num][loop.index0]+4)\n                        }}{{ir_line }}\x1b[41m{{ir_line_type-}}\x1b[39;49;00m\n                    {%- endfor -%}\n                {%- endif -%}\n            {%- endfor -%}\n    {%- endfor -%}\n    ')
    return ansi_template