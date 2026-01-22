import os
from typing import IO, Any, Dict, List, Sequence
from onnx import AttributeProto, defs, load
from onnx.backend.test.case import collect_snippets
from onnx.backend.test.loader import load_model_tests
from onnx.backend.test.runner import Runner
def gen_model_test_coverage(schemas: Sequence[defs.OpSchema], f: IO[Any], ml: bool) -> None:
    f.write('# Model Test Coverage\n')
    schema_dict = {}
    for schema in schemas:
        schema_dict[schema.name] = schema
    attrs: Dict[str, Dict[str, List[Any]]] = {}
    model_paths: List[Any] = []
    for rt in load_model_tests(kind='real'):
        if rt.url.startswith('onnx/backend/test/data/light/'):
            model_name = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', rt.url))
            if not os.path.exists(model_name):
                raise FileNotFoundError(f'Unable to find model {model_name!r}.')
            model_paths.append(model_name)
        else:
            model_dir = Runner.prepare_model_data(rt)
            model_paths.append(os.path.join(model_dir, 'model.onnx'))
    model_paths.sort()
    model_written = False
    for model_pb_path in model_paths:
        model = load(model_pb_path)
        if ml:
            ml_present = False
            for opset in model.opset_import:
                if opset.domain == 'ai.onnx.ml':
                    ml_present = True
            if not ml_present:
                continue
            else:
                model_written = True
        f.write(f'## {model.graph.name}\n')
        num_covered = 0
        for node in model.graph.node:
            if node.op_type in common_covered or node.op_type in experimental_covered:
                num_covered += 1
                for attr in node.attribute:
                    if node.op_type not in attrs:
                        attrs[node.op_type] = {}
                    if attr.name not in attrs[node.op_type]:
                        attrs[node.op_type][attr.name] = []
                    if attr.type == AttributeProto.FLOAT:
                        if attr.f not in attrs[node.op_type][attr.name]:
                            attrs[node.op_type][attr.name].append(attr.f)
                    elif attr.type == AttributeProto.INT:
                        if attr.i not in attrs[node.op_type][attr.name]:
                            attrs[node.op_type][attr.name].append(attr.i)
                    elif attr.type == AttributeProto.STRING:
                        if attr.s not in attrs[node.op_type][attr.name]:
                            attrs[node.op_type][attr.name].append(attr.s)
                    elif attr.type == AttributeProto.TENSOR:
                        if attr.t not in attrs[node.op_type][attr.name]:
                            attrs[node.op_type][attr.name].append(attr.t)
                    elif attr.type == AttributeProto.GRAPH:
                        if attr.g not in attrs[node.op_type][attr.name]:
                            attrs[node.op_type][attr.name].append(attr.g)
                    elif attr.type == AttributeProto.FLOATS:
                        if attr.floats not in attrs[node.op_type][attr.name]:
                            attrs[node.op_type][attr.name].append(attr.floats)
                    elif attr.type == AttributeProto.INTS:
                        if attr.ints not in attrs[node.op_type][attr.name]:
                            attrs[node.op_type][attr.name].append(attr.ints)
                    elif attr.type == AttributeProto.STRINGS:
                        if attr.strings not in attrs[node.op_type][attr.name]:
                            attrs[node.op_type][attr.name].append(attr.strings)
                    elif attr.type == AttributeProto.TENSORS:
                        if attr.tensors not in attrs[node.op_type][attr.name]:
                            attrs[node.op_type][attr.name].append(attr.tensors)
                    elif attr.type == AttributeProto.GRAPHS:
                        if attr.graphs not in attrs[node.op_type][attr.name]:
                            attrs[node.op_type][attr.name].append(attr.graphs)
        f.write(f'\n{model.graph.name} has {num_covered} nodes. Of these, {len(model.graph.node)} are covered by node tests ({100.0 * float(num_covered) / float(len(model.graph.node))}%)\n\n\n')
        f.write('<details>\n')
        f.write('<summary>nodes</summary>\n\n')
        for op in sorted(attrs):
            f.write('<details>\n')
            f.write(f'<summary>{op}: {len(attrs[op])} out of {len(schema_dict[op].attributes)} attributes covered</summary>\n\n')
            for attribute in sorted(schema_dict[op].attributes):
                if attribute in attrs[op]:
                    f.write(f'{attribute}: {len(attrs[op][attribute])}\n')
                else:
                    f.write(f'{attribute}: 0\n')
            f.write('</details>\n')
        f.write('</details>\n\n\n')
    if not model_written and ml:
        f.write('No model tests present for selected domain\n')