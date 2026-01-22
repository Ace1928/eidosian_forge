import re
import winreg
def stringdisplay(s):
    """convert "d.d.d.d,d.d.d.d" to ["d.d.d.d","d.d.d.d"].
       also handle u'd.d.d.d d.d.d.d', as reporting on SF 
    """
    import re
    return list(map(str, re.split('[ ,]', s)))