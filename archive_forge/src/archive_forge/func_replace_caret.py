import logging
import re
def replace_caret(comp, loose):
    r = regexp[CARETLOOSE] if loose else regexp[CARET]

    def repl(mob):
        m0 = mob.group(0)
        M, m, p, pr, _ = mob.groups()
        logger.debug('caret %s %s %s %s %s %s', comp, m0, M, m, p, pr)
        if is_x(M):
            ret = ''
        elif is_x(m):
            ret = '>=' + M + '.0.0 <' + str(int(M) + 1) + '.0.0'
        elif is_x(p):
            if M == '0':
                ret = '>=' + M + '.' + m + '.0 <' + M + '.' + str(int(m) + 1) + '.0'
            else:
                ret = '>=' + M + '.' + m + '.0 <' + str(int(M) + 1) + '.0.0'
        elif pr:
            logger.debug('replaceCaret pr %s', pr)
            if pr[0] != '-':
                pr = '-' + pr
            if M == '0':
                if m == '0':
                    ret = '>=' + M + '.' + m + '.' + (p or '') + pr + ' <' + M + '.' + m + '.' + str(int(p or 0) + 1)
                else:
                    ret = '>=' + M + '.' + m + '.' + (p or '') + pr + ' <' + M + '.' + str(int(m) + 1) + '.0'
            else:
                ret = '>=' + M + '.' + m + '.' + (p or '') + pr + ' <' + str(int(M) + 1) + '.0.0'
        elif M == '0':
            if m == '0':
                ret = '>=' + M + '.' + m + '.' + (p or '') + ' <' + M + '.' + m + '.' + str(int(p or 0) + 1)
            else:
                ret = '>=' + M + '.' + m + '.' + (p or '') + ' <' + M + '.' + str(int(m) + 1) + '.0'
        else:
            ret = '>=' + M + '.' + m + '.' + (p or '') + ' <' + str(int(M) + 1) + '.0.0'
        logger.debug('caret return %s', ret)
        return ret
    return r.sub(repl, comp)