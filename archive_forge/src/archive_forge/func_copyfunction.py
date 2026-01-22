from types import CodeType as code, FunctionType as function
def copyfunction(template, funcchanges, codechanges):
    names = ['globals', 'name', 'defaults', 'closure']
    values = [funcchanges.get(name, getattr(template, '__' + name + '__')) for name in names]
    return function(copycode(template.__code__, codechanges), *values)