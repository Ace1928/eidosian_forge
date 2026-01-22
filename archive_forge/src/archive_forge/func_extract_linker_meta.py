from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union
from dataclasses import dataclass
def extract_linker_meta(self, header: str):
    for ln in header.splitlines():
        if ln.startswith('//'):
            m = self.linker_directives.match(ln)
            if _exists(m):
                ker_name, c_sig, algo_info = (m.group(1), m.group(2), m.group(3))
                name, sig_hash, suffix = self._match_name(ker_name)
                c_types, arg_names = self._match_c_sig(c_sig)
                num_specs, sizes = self._match_suffix(suffix, c_sig)
                self._add_kernel('_'.join([name, algo_info]), KernelLinkerMeta(orig_kernel_name=name, arg_names=arg_names, arg_ctypes=c_types, sizes=sizes, sig_hash=sig_hash, triton_suffix=suffix, suffix=suffix, num_specs=num_specs))