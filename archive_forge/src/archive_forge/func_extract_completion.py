def extract_completion(var_type):
    s = subprocess.Popen(['scilab', '-nwni'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = s.communicate('fd = mopen("/dev/stderr", "wt");\nmputl(strcat(completion("", "%s"), "||"), fd);\nmclose(fd)\n' % var_type)
    if '||' not in output[1]:
        raise Exception(output[0])
    text = output[1].strip()
    if text.startswith('Error: unable to open display \n'):
        text = text[len('Error: unable to open display \n'):]
    return text.split('||')