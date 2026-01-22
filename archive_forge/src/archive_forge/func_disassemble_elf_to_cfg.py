from tempfile import NamedTemporaryFile, TemporaryDirectory
import os
import warnings
from numba.core.errors import NumbaWarning
def disassemble_elf_to_cfg(elf, mangled_symbol):
    """
    Gets the CFG of the disassembly of an ELF object, elf, at mangled name,
    mangled_symbol, and renders it appropriately depending on the execution
    environment (terminal/notebook).
    """
    try:
        import r2pipe
    except ImportError:
        raise RuntimeError('r2pipe package needed for disasm CFG')

    def get_rendering(cmd=None):
        from numba.pycc.platform import Toolchain
        if cmd is None:
            raise ValueError('No command given')
        with TemporaryDirectory() as tmpdir:
            with NamedTemporaryFile(delete=False, dir=tmpdir) as f:
                f.write(elf)
                f.flush()
            linked = False
            try:
                raw_dso_name = f'{os.path.basename(f.name)}.so'
                linked_dso = os.path.join(tmpdir, raw_dso_name)
                tc = Toolchain()
                tc.link_shared(linked_dso, (f.name,))
                obj_to_analyse = linked_dso
                linked = True
            except Exception as e:
                msg = f'Linking the ELF object with the distutils toolchain failed with: {e}. Disassembly will still work but might be less accurate and will not use DWARF information.'
                warnings.warn(NumbaWarning(msg))
                obj_to_analyse = f.name
            try:
                flags = ['-2', '-e io.cache=true', '-e scr.color=1', '-e asm.dwarf=true', '-e scr.utf8=true']
                r = r2pipe.open(obj_to_analyse, flags=flags)
                r.cmd('aaaaaa')
                if linked:
                    mangled_symbol_61char = mangled_symbol[:61]
                    r.cmd('e bin.demangle=false')
                    r.cmd(f's `is~ {mangled_symbol_61char}[1]`')
                    r.cmd('e bin.demangle=true')
                data = r.cmd('%s' % cmd)
                r.quit()
            except Exception as e:
                if 'radare2 in PATH' in str(e):
                    msg = "This feature requires 'radare2' to be installed and available on the system see: https://github.com/radareorg/radare2. Cannot find 'radare2' in $PATH."
                    raise RuntimeError(msg)
                else:
                    raise e
        return data

    class DisasmCFG(object):

        def _repr_svg_(self):
            try:
                import graphviz
            except ImportError:
                raise RuntimeError('graphviz package needed for disasm CFG')
            jupyter_rendering = get_rendering(cmd='agfd')
            jupyter_rendering.replace('fontname="Courier",', 'fontname="Courier",fontsize=6,')
            src = graphviz.Source(jupyter_rendering)
            return src.pipe('svg').decode('UTF-8')

        def __repr__(self):
            return get_rendering(cmd='agf')
    return DisasmCFG()