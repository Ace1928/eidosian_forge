import threading, inspect, shlex
def print_usage(self):
    line_width = 100
    outArr = []

    def addToOut(ind, cmd):
        if ind >= len(outArr):
            outArr.extend([None] * (ind - len(outArr) + 1))
        if outArr[ind] != None:
            for i in range(len(outArr) - 1, 0, -1):
                if outArr[i] is None:
                    outArr[i] = outArr[ind]
                    outArr[ind] = cmd
                    return
            outArr.append(cmd)
        else:
            outArr[ind] = cmd
    for cmd, subcommands in self.commands.items():
        for subcmd, subcmdDetails in subcommands.items():
            out = ''
            out += ('/%s ' % cmd).ljust(15)
            out += ('%s ' % subcmd if subcmd != '_' else '').ljust(15)
            args = '%s ' % ' '.join(['<%s>' % c for c in subcmdDetails['args'][0:len(subcmdDetails['args']) - subcmdDetails['optional']]])
            args += '%s ' % ' '.join(['[%s]' % c for c in subcmdDetails['args'][len(subcmdDetails['args']) - subcmdDetails['optional']:]])
            out += args.ljust(30)
            out += subcmdDetails['desc'].ljust(20)
            addToOut(subcmdDetails['order'], out)
    print('----------------------------------------------')
    print('\n'.join(outArr))
    print('----------------------------------------------')