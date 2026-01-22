import sys
def get_pipe_name(name):
    name = name.replace('/', '')
    name = name.replace('\\', '')
    name = '\\\\.\\pipe\\' + name
    return name