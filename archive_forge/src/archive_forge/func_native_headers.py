from __future__ import annotations
import pathlib
import typing as T
from mesonbuild import mesonlib
from mesonbuild.build import CustomTarget, CustomTargetIndex, GeneratedList, Target
from mesonbuild.compilers import detect_compiler_for
from mesonbuild.interpreterbase.decorators import ContainerTypeInfo, FeatureDeprecated, FeatureNew, KwargInfo, typed_pos_args, typed_kwargs
from mesonbuild.mesonlib import version_compare, MachineChoice
from . import NewExtensionModule, ModuleReturnValue, ModuleInfo
from ..interpreter.type_checking import NoneType
@FeatureNew('java.native_headers', '1.0.0')
@typed_pos_args('java.native_headers', varargs=(str, mesonlib.File, Target, CustomTargetIndex, GeneratedList))
@typed_kwargs('java.native_headers', KwargInfo('classes', ContainerTypeInfo(list, str), default=[], listify=True, required=True), KwargInfo('package', (str, NoneType), default=None))
def native_headers(self, state: ModuleState, args: T.Tuple[T.List[mesonlib.FileOrString]], kwargs: T.Dict[str, T.Optional[str]]) -> ModuleReturnValue:
    return self.__native_headers(state, args, kwargs)