from __future__ import annotations
import os
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleInfo
from ..build import CustomTarget
from ..interpreter.type_checking import NoneType, in_set_validator
from ..interpreterbase import typed_pos_args, typed_kwargs, KwargInfo
from ..mesonlib import File, MesonException
@typed_pos_args('wayland.scan_xml', varargs=(str, File), min_varargs=1)
@typed_kwargs('wayland.scan_xml', KwargInfo('public', bool, default=False), KwargInfo('client', bool, default=True), KwargInfo('server', bool, default=False), KwargInfo('include_core_only', bool, default=True, since='0.64.0'))
def scan_xml(self, state: ModuleState, args: T.Tuple[T.List[FileOrString]], kwargs: ScanXML) -> ModuleReturnValue:
    if self.scanner_bin is None:
        dep = state.dependency('wayland-client')
        self.scanner_bin = state.find_tool('wayland-scanner', 'wayland-scanner', 'wayland_scanner', wanted=dep.version)
    scope = 'public' if kwargs['public'] else 'private'
    sides = [i for i in T.cast("T.List[Literal['client', 'server']]", ['client', 'server']) if kwargs[i]]
    if not sides:
        raise MesonException('At least one of client or server keyword argument must be set to true.')
    xml_files = self.interpreter.source_strings_to_files(args[0])
    targets: T.List[CustomTarget] = []
    for xml_file in xml_files:
        name = os.path.splitext(os.path.basename(xml_file.fname))[0]
        code = CustomTarget(f'{name}-protocol', state.subdir, state.subproject, state.environment, [self.scanner_bin, f'{scope}-code', '@INPUT@', '@OUTPUT@'], [xml_file], [f'{name}-protocol.c'], backend=state.backend)
        targets.append(code)
        for side in sides:
            command = [self.scanner_bin, f'{side}-header', '@INPUT@', '@OUTPUT@']
            if kwargs['include_core_only']:
                command.append('--include-core-only')
            header = CustomTarget(f'{name}-{side}-protocol', state.subdir, state.subproject, state.environment, command, [xml_file], [f'{name}-{side}-protocol.h'], backend=state.backend)
            targets.append(header)
    return ModuleReturnValue(targets, targets)