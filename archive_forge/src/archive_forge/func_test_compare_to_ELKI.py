import warnings
import numpy as np
import pytest
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.cluster._optics import _extend_region, _extract_xi_labels
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def test_compare_to_ELKI():
    r1 = [np.inf, 1.0574896366427478, 0.7587934993548423, 0.7290174038973836, 0.7290174038973836, 0.7290174038973836, 0.6861627576116127, 0.7587934993548423, 0.9280118450166668, 1.1748022534146194, 3.3355455741292257, 0.49618389254482587, 0.2552805046961355, 0.2552805046961355, 0.24944622248445714, 0.24944622248445714, 0.24944622248445714, 0.2552805046961355, 0.2552805046961355, 0.3086779122185853, 4.163024452756142, 1.623152630340929, 0.45315840475822655, 0.25468325192031926, 0.2254004358159971, 0.18765711877083036, 0.1821471333893275, 0.1821471333893275, 0.18765711877083036, 0.18765711877083036, 0.2240202988740153, 1.154337614548715, 1.342604473837069, 1.323308536402633, 0.8607514948648837, 0.27219111215810565, 0.13260875220533205, 0.13260875220533205, 0.09890587675958984, 0.09890587675958984, 0.13548790801634494, 0.1575483940837384, 0.17515137170530226, 0.17575920159442388, 0.27219111215810565, 0.6101447895405373, 1.3189208094864302, 1.323308536402633, 2.2509184159764577, 2.4517810628594527, 3.675977064404973, 3.8264795626020365, 2.9130735341510614, 2.9130735341510614, 2.9130735341510614, 2.9130735341510614, 2.8459300127258036, 2.8459300127258036, 2.8459300127258036, 3.0321982337972537]
    o1 = [0, 3, 6, 4, 7, 8, 2, 9, 5, 1, 31, 30, 32, 34, 33, 38, 39, 35, 37, 36, 44, 21, 23, 24, 22, 25, 27, 29, 26, 28, 20, 40, 45, 46, 10, 15, 11, 13, 17, 19, 18, 12, 16, 14, 47, 49, 43, 48, 42, 41, 53, 57, 51, 52, 56, 59, 54, 55, 58, 50]
    p1 = [-1, 0, 3, 6, 6, 6, 8, 3, 7, 5, 1, 31, 30, 30, 34, 34, 34, 32, 32, 37, 36, 44, 21, 23, 24, 22, 25, 25, 22, 22, 22, 21, 40, 45, 46, 10, 15, 15, 13, 13, 15, 11, 19, 15, 10, 47, 12, 45, 14, 43, 42, 53, 57, 57, 57, 57, 59, 59, 59, 58]
    clust1 = OPTICS(min_samples=5).fit(X)
    assert_array_equal(clust1.ordering_, np.array(o1))
    assert_array_equal(clust1.predecessor_[clust1.ordering_], np.array(p1))
    assert_allclose(clust1.reachability_[clust1.ordering_], np.array(r1))
    for i in clust1.ordering_[1:]:
        assert clust1.reachability_[i] >= clust1.core_distances_[clust1.predecessor_[i]]
    r2 = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 0.27219111215810565, 0.13260875220533205, 0.13260875220533205, 0.09890587675958984, 0.09890587675958984, 0.13548790801634494, 0.1575483940837384, 0.17515137170530226, 0.17575920159442388, 0.27219111215810565, 0.4928068613197889, np.inf, 0.2666183922512113, 0.18765711877083036, 0.1821471333893275, 0.1821471333893275, 0.1821471333893275, 0.18715928772277457, 0.18765711877083036, 0.18765711877083036, 0.25468325192031926, np.inf, 0.2552805046961355, 0.2552805046961355, 0.24944622248445714, 0.24944622248445714, 0.24944622248445714, 0.2552805046961355, 0.2552805046961355, 0.3086779122185853, 0.34466409325984865, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
    o2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 11, 13, 17, 19, 18, 12, 16, 14, 47, 46, 20, 22, 25, 23, 27, 29, 24, 26, 28, 21, 30, 32, 34, 33, 38, 39, 35, 37, 36, 31, 40, 41, 42, 43, 44, 45, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    p2 = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10, 15, 15, 13, 13, 15, 11, 19, 15, 10, 47, -1, 20, 22, 25, 25, 25, 25, 22, 22, 23, -1, 30, 30, 34, 34, 34, 32, 32, 37, 38, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    clust2 = OPTICS(min_samples=5, max_eps=0.5).fit(X)
    assert_array_equal(clust2.ordering_, np.array(o2))
    assert_array_equal(clust2.predecessor_[clust2.ordering_], np.array(p2))
    assert_allclose(clust2.reachability_[clust2.ordering_], np.array(r2))
    index = np.where(clust1.core_distances_ <= 0.5)[0]
    assert_allclose(clust1.core_distances_[index], clust2.core_distances_[index])