import requests
def create_unexpected_kwargs_error(name, kwargs):
    quoted_kwargs = [f"'{k}'" for k in sorted(kwargs)]
    text = [f'{name}() ']
    if len(quoted_kwargs) == 1:
        text.append('got an unexpected keyword argument ')
    else:
        text.append('got unexpected keyword arguments ')
    text.append(', '.join(quoted_kwargs))
    return TypeError(''.join(text))