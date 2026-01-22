import codecs
import itertools
import sys
from enum import Enum, auto
from typing import Optional, List, Sequence, Union
from .termhelpers import Termmode
from .curtsieskeys import CURTSIES_NAMES as special_curtsies_names
def try_keys() -> None:
    print('press a bunch of keys (not at the same time, but you can hit them pretty quickly)')
    import os
    from .termhelpers import Cbreak

    def ask_what_they_pressed(seq: bytes, Normal: Termmode) -> None:
        print('Unidentified character sequence!')
        with Normal:
            while True:
                r = input("type 'ok' to prove you're not pounding keys ")
                if r.lower().strip() == 'ok':
                    break
        while True:
            print(f'Press the key that produced {seq!r} again please')
            retry = os.read(sys.stdin.fileno(), 1000)
            if seq == retry:
                break
            print("nope, that wasn't it")
        with Normal:
            name = input('Describe in English what key you pressed: ')
            f = open('keylog.txt', 'a')
            f.write(f'{seq!r} is called {name}\n')
            f.close()
            print('Thanks! Please open an issue at https://github.com/bpython/curtsies/issues')
            print('or email thomasballinger@gmail.com. Include this terminal history or keylog.txt.')
            print('You can keep pressing keys')
    with Cbreak(sys.stdin) as NoCbreak:
        while True:
            try:
                chars = os.read(sys.stdin.fileno(), 1000)
                print('---')
                print(repr(chars))
                if chars in CURTSIES_NAMES:
                    print(CURTSIES_NAMES[chars])
                elif len(chars) == 1:
                    print('literal')
                else:
                    print('unknown!!!')
                    ask_what_they_pressed(chars, NoCbreak)
            except OSError:
                pass