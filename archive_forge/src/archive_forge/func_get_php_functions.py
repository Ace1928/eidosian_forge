from __future__ import print_function
def get_php_functions():
    function_re = re.compile(PHP_FUNCTION_RE)
    module_re = re.compile(PHP_MODULE_RE)
    modules = {}
    for file in get_php_references():
        module = ''
        for line in open(file):
            if not module:
                search = module_re.search(line)
                if search:
                    module = search.group(1)
                    modules[module] = []
            elif 'href="function.' in line:
                for match in function_re.finditer(line):
                    fn = match.group(1)
                    if '-&gt;' not in fn and '::' not in fn and (fn not in modules[module]):
                        modules[module].append(fn)
        if module:
            if module == 'PHP Options/Info':
                modules[module].remove('main')
            if module == 'Filesystem':
                modules[module].remove('delete')
            if not modules[module]:
                del modules[module]
    return modules