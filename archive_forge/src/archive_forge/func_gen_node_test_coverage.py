import os
from typing import IO, Any, Dict, List, Sequence
from onnx import AttributeProto, defs, load
from onnx.backend.test.case import collect_snippets
from onnx.backend.test.loader import load_model_tests
from onnx.backend.test.runner import Runner
def gen_node_test_coverage(schemas: Sequence[defs.OpSchema], f: IO[Any], ml: bool) -> None:
    global common_covered
    global experimental_covered
    generators = set({'Multinomial', 'RandomNormal', 'RandomNormalLike', 'RandomUniform', 'RandomUniformLike'})
    node_tests = collect_snippets()
    common_covered = sorted((s.name for s in schemas if s.name in node_tests and s.support_level == defs.OpSchema.SupportType.COMMON and ((s.domain == 'ai.onnx.ml') == ml)))
    common_no_cover = sorted((s.name for s in schemas if s.name not in node_tests and s.support_level == defs.OpSchema.SupportType.COMMON and ((s.domain == 'ai.onnx.ml') == ml)))
    common_generator = sorted((name for name in common_no_cover if name in generators))
    experimental_covered = sorted((s.name for s in schemas if s.name in node_tests and s.support_level == defs.OpSchema.SupportType.EXPERIMENTAL and ((s.domain == 'ai.onnx.ml') == ml)))
    experimental_no_cover = sorted((s.name for s in schemas if s.name not in node_tests and s.support_level == defs.OpSchema.SupportType.EXPERIMENTAL and ((s.domain == 'ai.onnx.ml') == ml)))
    experimental_generator = sorted((name for name in experimental_no_cover if name in generators))
    num_common = len(common_covered) + len(common_no_cover) - len(common_generator)
    num_experimental = len(experimental_covered) + len(experimental_no_cover) - len(experimental_generator)
    f.write('# Node Test Coverage\n')
    f.write('## Summary\n')
    if num_common:
        f.write(f'Node tests have covered {len(common_covered)}/{num_common} ({len(common_covered) / float(num_common) * 100:.2f}%, {len(common_generator)} generators excluded) common operators.\n\n')
    else:
        f.write('Node tests have covered 0/0 (N/A) common operators. \n\n')
    if num_experimental:
        f.write('Node tests have covered {}/{} ({:.2f}%, {} generators excluded) experimental operators.\n\n'.format(len(experimental_covered), num_experimental, len(experimental_covered) / float(num_experimental) * 100, len(experimental_generator)))
    else:
        f.write('Node tests have covered 0/0 (N/A) experimental operators.\n\n')
    titles = ['&#x1F49A;Covered Common Operators', '&#x1F494;No Cover Common Operators', '&#x1F49A;Covered Experimental Operators', '&#x1F494;No Cover Experimental Operators']
    all_lists = [common_covered, common_no_cover, experimental_covered, experimental_no_cover]
    for t in titles:
        f.write(f'* [{t[9:]}](#{t[9:].lower().replace(' ', '-')})\n')
    f.write('\n')
    for t, l in zip(titles, all_lists):
        f.write(f'## {t}\n')
        for s in l:
            f.write(f'### {s}')
            if s in node_tests:
                f.write(f'\nThere are {len(node_tests[s])} test cases, listed as following:\n')
                for summary, code in sorted(node_tests[s]):
                    f.write('<details>\n')
                    f.write(f'<summary>{summary}</summary>\n\n')
                    f.write(f'```python\n{code}\n```\n\n')
                    f.write('</details>\n')
            elif s in generators:
                f.write(' (random generator operator)\n')
            else:
                f.write(' (call for test cases)\n')
            f.write('\n\n')
        f.write('<br/>\n\n')