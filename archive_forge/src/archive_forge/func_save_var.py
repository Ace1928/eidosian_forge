from typing import Dict, List, Sequence, Tuple
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.code_template import CodeTemplate
from torchgen.model import Argument, FunctionSchema
from torchgen.utils import FileManager
from .gen_inplace_or_view_type import VIEW_FUNCTIONS
def save_var(var: SavedAttribute, is_output: bool) -> None:
    name = var.nctype.name
    type = var.nctype.type
    should_append_getsetdef = True
    should_append_raw_getsetdef = False
    visit_name = name
    if type == BaseCType(tensorT) or type == OptionalCType(BaseCType(tensorT)) or type == MutRefCType(OptionalCType(BaseCType(tensorT))) or (type == BaseCType(scalarT) and is_output):
        saved_variables.append(f'SavedVariable {name}_;')
        release_variables.append(f'{name}_.reset_data();')
        ptr = 'shared_from_this()' if is_output else ''
        unpack.append(f'auto {name} = {name}_.unpack({ptr});')
        getter_definitions.append(GETTER_DEFINITION_SAVEDVAR.substitute(op=info.op, name=name, body=GETTER_BODY_SAVEDVAR))
        getter_definitions.append(GETTER_DEFINITION_RAW_SAVEDVAR.substitute(op=info.op, name=name, body=GETTER_BODY_RAW_SAVEDVAR))
        should_append_raw_getsetdef = True
        visit_name = f'{name}_'
    elif type == BaseCType(tensorListT) or type == BaseCType(iTensorListRefT) or type == VectorCType(BaseCType(tensorT)):
        if type == VectorCType(BaseCType(tensorT)):
            assert info.func.func.name.name.base.startswith('_foreach') and is_output
        saved_variables.append(f'std::vector<SavedVariable> {name}_;')
        saved_variables.append(f'bool {name}_released_ = false;')
        release_variables.append(f'{name}_.clear();')
        release_variables.append(f'{name}_released_ = true;')
        ptr = 'shared_from_this()' if is_output else 'nullptr'
        unpack.append(f'auto {name} = unpack_list({name}_, {ptr});')
        asserts.append(f'TORCH_CHECK(!{name}_released_, ERR_BACKWARD_TWICE);')
        getter_definitions.append(GETTER_DEFINITION_VEC_SAVEDVAR.substitute(op=info.op, name=name, body=GETTER_BODY_VEC_SAVEDVAR))
        getter_definitions.append(GETTER_DEFINITION_RAW_VEC_SAVEDVAR.substitute(op=info.op, name=name, body=GETTER_BODY_RAW_VEC_SAVEDVAR))
        should_append_raw_getsetdef = True
        visit_name = f'{name}_'
    elif type == ListCType(OptionalCType(BaseCType(tensorT))):
        saved_variables.append(f'std::vector<SavedVariable> {name}_;')
        saved_variables.append(f'bool {name}_released_ = false;')
        release_variables.append(f'{name}_.clear();')
        release_variables.append(f'{name}_released_ = true;')
        unpack.append(f'auto {name} = unpack_opt_list({name}_);')
        asserts.append(f'TORCH_CHECK(!{name}_released_, ERR_BACKWARD_TWICE);')
        getter_definitions.append(GETTER_DEFINITION_VEC_SAVEDVAR.substitute(op=info.op, name=name, body=GETTER_BODY_VEC_SAVEDVAR))
        getter_definitions.append(GETTER_DEFINITION_RAW_VEC_SAVEDVAR.substitute(op=info.op, name=name, body=GETTER_BODY_RAW_VEC_SAVEDVAR))
        should_append_raw_getsetdef = True
        visit_name = f'{name}_'
    elif type == BaseCType(intArrayRefT):
        saved_variables.append(f'std::vector<int64_t> {name};')
        getter_definitions.append(GETTER_DEFINITION.substitute(op=info.op, name=name, body=GETTER_BODY_ARRAYREF_LONG))
    elif type == BaseCType(symIntArrayRefT):
        saved_variables.append(f'std::vector<c10::SymInt> {name};')
        getter_definitions.append(GETTER_DEFINITION.substitute(op=info.op, name=name, body=GETTER_BODY_ARRAYREF_SYMINT))
    elif type == BaseCType(optionalIntArrayRefT):
        saved_variables.append(f'c10::OptionalArray<int64_t> {name};')
        getter_definitions.append(GETTER_DEFINITION_OPT_ARRAYREF.substitute(op=info.op, name=name, body=GETTER_BODY_ARRAYREF_LONG))
    elif type == BaseCType(optionalSymIntArrayRefT):
        saved_variables.append(f'c10::OptionalArray<c10::SymInt> {name};')
        getter_definitions.append(GETTER_DEFINITION_OPT_ARRAYREF.substitute(op=info.op, name=name, body=GETTER_BODY_ARRAYREF_SYMINT))
    elif type == OptionalCType(BaseCType(intArrayRefT)):
        saved_variables.append(f'c10::OptionalArray<int64_t> {name};')
        getter_definitions.append(GETTER_DEFINITION_OPT_ARRAYREF.substitute(op=info.op, name=name, body=GETTER_BODY_ARRAYREF_LONG))
    elif type == OptionalCType(BaseCType(symIntArrayRefT)):
        saved_variables.append(f'c10::OptionalArray<c10::SymInt> {name};')
        getter_definitions.append(GETTER_DEFINITION_OPT_ARRAYREF.substitute(op=info.op, name=name, body=GETTER_BODY_ARRAYREF_SYMINT))
    elif type == OptionalCType(ArrayRefCType(BaseCType(doubleT))):
        saved_variables.append(f'c10::OptionalArray<double> {name};')
        getter_definitions.append(GETTER_DEFINITION_OPT_ARRAYREF.substitute(op=info.op, name=name, body=GETTER_BODY_ARRAYREF_DOUBLE))
    elif type == BaseCType(longT):
        saved_variables.append(f'{type.cpp_type()} {name} = 0;')
        getter_definitions.append(GETTER_DEFINITION.substitute(op=info.op, name=name, body=GETTER_BODY_INT64_T))
    elif type == BaseCType(SymIntT):
        saved_variables.append(f'c10::SymInt {name};')
        getter_definitions.append(GETTER_DEFINITION.substitute(op=info.op, name=name, body=GETTER_BODY_SYMINT))
    elif type == BaseCType(stringT):
        saved_variables.append(f'std::string {name};')
        getter_definitions.append(GETTER_DEFINITION.substitute(op=info.op, name=name, body=GETTER_BODY_STRING))
    elif type == OptionalCType(BaseCType(stringT)):
        saved_variables.append(f'c10::optional<std::string> {name};')
        getter_definitions.append(GETTER_DEFINITION_OPT.substitute(op=info.op, name=name, body=GETTER_BODY_STRING))
    elif type == ArrayRefCType(elem=BaseCType(type=BaseCppType(ns='at', name='Scalar'))):
        saved_variables.append(f'std::vector<at::Scalar> {name};')
        saved_variables.append(f'bool {name}_released_ = false;')
        release_variables.append(f'{name}.clear();')
        getter_definitions.append(CodeTemplate('PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {\n  HANDLE_TH_ERRORS\n  const auto *node = static_cast<${op}*>(self->cdata.get());\n  const auto& prop = node->${name};\n  if (node->${name}_released_) {\n    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);\n    return nullptr;\n  }\n  ${body}\n  END_HANDLE_TH_ERRORS\n}\n                            ').substitute(op=info.op, name=name, body=GETTER_BODY_VEC_SCALAR))
    else:
        assert 'ref' not in type.cpp_type().lower() and 'view' not in type.cpp_type().lower() and ('*' not in type.cpp_type()) and ('&' not in type.cpp_type()), f'{type.cpp_type()} looks like it contains a non-owning reference'
        saved_variables.append(f'{type.cpp_type()} {name};')
        if type in MISC_GETTER_DEFS:
            getter_def, body = MISC_GETTER_DEFS[type]
            getter_definitions.append(getter_def.substitute(op=info.op, name=name, body=body))
        else:
            should_append_getsetdef = False
    if should_append_getsetdef:
        py_getsetdef_structs.append(PY_GETSETDEF_STRUCT.substitute(op=info.op, name=name))
    if should_append_raw_getsetdef:
        py_getsetdef_structs.append(PY_RAW_GETSETDEF_STRUCT.substitute(op=info.op, name=name))
    compiled_args.append(f'args.collect({visit_name});')
    apply_with_saved_before.append(f'saved.before({visit_name});')
    apply_with_saved_after.append(f'saved.after({visit_name});')