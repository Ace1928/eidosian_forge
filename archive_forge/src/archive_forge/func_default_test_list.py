from breezy import pyutils, transport
from breezy.bzr.vf_repository import InterDifferingSerializer
from breezy.errors import UninitializableFormat
from breezy.repository import InterRepository, format_registry
from breezy.tests import TestSkipped, default_transport, multiply_tests
from breezy.tests.per_controldir.test_controldir import TestCaseWithControlDir
from breezy.transport import FileExists
def default_test_list():
    """Generate the default list of interrepo permutations to test."""
    from breezy.bzr import groupcompress_repo, knitpack_repo, knitrepo
    result = []

    def add_combo(interrepo_cls, from_format, to_format, extra_setup=None, label=None):
        if label is None:
            label = interrepo_cls.__name__
        result.append((label, from_format, to_format, extra_setup))
    for optimiser_class in InterRepository.iter_optimisers():
        format_to_test = optimiser_class._get_repo_format_to_test()
        if format_to_test is not None:
            add_combo(optimiser_class, format_to_test, format_to_test)

    def force_known_graph(testcase):
        from breezy.bzr.fetch import Inter1and2Helper
        testcase.overrideAttr(Inter1and2Helper, 'known_graph_threshold', -1)
    for module_name in format_registry._get_all_modules():
        module = pyutils.get_named_object(module_name)
        try:
            get_extra_interrepo_test_combinations = getattr(module, 'get_extra_interrepo_test_combinations')
        except AttributeError:
            continue
        for interrepo_cls, from_format, to_format in get_extra_interrepo_test_combinations():
            add_combo(interrepo_cls, from_format, to_format)
    add_combo(InterRepository, knitrepo.RepositoryFormatKnit1(), knitrepo.RepositoryFormatKnit3())
    add_combo(knitrepo.InterKnitRepo, knitrepo.RepositoryFormatKnit1(), knitpack_repo.RepositoryFormatKnitPack1())
    add_combo(knitrepo.InterKnitRepo, knitpack_repo.RepositoryFormatKnitPack1(), knitrepo.RepositoryFormatKnit1())
    add_combo(knitrepo.InterKnitRepo, knitrepo.RepositoryFormatKnit3(), knitpack_repo.RepositoryFormatKnitPack3())
    add_combo(knitrepo.InterKnitRepo, knitpack_repo.RepositoryFormatKnitPack3(), knitrepo.RepositoryFormatKnit3())
    add_combo(knitrepo.InterKnitRepo, knitpack_repo.RepositoryFormatKnitPack3(), knitpack_repo.RepositoryFormatKnitPack4())
    add_combo(InterDifferingSerializer, knitpack_repo.RepositoryFormatKnitPack1(), knitpack_repo.RepositoryFormatKnitPack6RichRoot())
    add_combo(InterDifferingSerializer, knitpack_repo.RepositoryFormatKnitPack1(), knitpack_repo.RepositoryFormatKnitPack6RichRoot(), force_known_graph, label='InterDifferingSerializer+get_known_graph_ancestry')
    add_combo(InterDifferingSerializer, knitpack_repo.RepositoryFormatKnitPack6RichRoot(), groupcompress_repo.RepositoryFormat2a())
    add_combo(InterDifferingSerializer, groupcompress_repo.RepositoryFormat2a(), knitpack_repo.RepositoryFormatKnitPack6RichRoot())
    return result