import logging
import re
def replace_tilde(comp, loose):
    r = regexp[TILDELOOSE] if loose else regexp[TILDE]

    def repl(mob):
        _ = mob.group(0)
        M, m, p, pr, _ = mob.groups()
        logger.debug('tilde %s %s %s %s %s %s', comp, _, M, m, p, pr)
        if is_x(M):
            ret = ''
        elif is_x(m):
            ret = '>=' + M + '.0.0 <' + str(int(M) + 1) + '.0.0'
        elif is_x(p):
            ret = '>=' + M + '.' + m + '.0 <' + M + '.' + str(int(m) + 1) + '.0'
        elif pr:
            logger.debug('replaceTilde pr %s', pr)
            if pr[0] != '-':
                pr = '-' + pr
            ret = '>=' + M + '.' + m + '.' + p + pr + ' <' + M + '.' + str(int(m) + 1) + '.0'
        else:
            ret = '>=' + M + '.' + m + '.' + p + ' <' + M + '.' + str(int(m) + 1) + '.0'
        logger.debug('tilde return, %s', ret)
        return ret
    return r.sub(repl, comp)