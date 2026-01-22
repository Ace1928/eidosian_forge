import operator
import warnings
def jaro_winkler_similarity(s1, s2, p=0.1, max_l=4):
    """
    The Jaro Winkler distance is an extension of the Jaro similarity in:

        William E. Winkler. 1990. String Comparator Metrics and Enhanced
        Decision Rules in the Fellegi-Sunter Model of Record Linkage.
        Proceedings of the Section on Survey Research Methods.
        American Statistical Association: 354-359.

    such that:

        jaro_winkler_sim = jaro_sim + ( l * p * (1 - jaro_sim) )

    where,

    - jaro_sim is the output from the Jaro Similarity,
        see jaro_similarity()
    - l is the length of common prefix at the start of the string
        - this implementation provides an upperbound for the l value
            to keep the prefixes.A common value of this upperbound is 4.
    - p is the constant scaling factor to overweigh common prefixes.
        The Jaro-Winkler similarity will fall within the [0, 1] bound,
        given that max(p)<=0.25 , default is p=0.1 in Winkler (1990)


    Test using outputs from https://www.census.gov/srd/papers/pdf/rr93-8.pdf
    from "Table 5 Comparison of String Comparators Rescaled between 0 and 1"

    >>> winkler_examples = [("billy", "billy"), ("billy", "bill"), ("billy", "blily"),
    ... ("massie", "massey"), ("yvette", "yevett"), ("billy", "bolly"), ("dwayne", "duane"),
    ... ("dixon", "dickson"), ("billy", "susan")]

    >>> winkler_scores = [1.000, 0.967, 0.947, 0.944, 0.911, 0.893, 0.858, 0.853, 0.000]
    >>> jaro_scores =    [1.000, 0.933, 0.933, 0.889, 0.889, 0.867, 0.822, 0.790, 0.000]

    One way to match the values on the Winkler's paper is to provide a different
    p scaling factor for different pairs of strings, e.g.

    >>> p_factors = [0.1, 0.125, 0.20, 0.125, 0.20, 0.20, 0.20, 0.15, 0.1]

    >>> for (s1, s2), jscore, wscore, p in zip(winkler_examples, jaro_scores, winkler_scores, p_factors):
    ...     assert round(jaro_similarity(s1, s2), 3) == jscore
    ...     assert round(jaro_winkler_similarity(s1, s2, p=p), 3) == wscore


    Test using outputs from https://www.census.gov/srd/papers/pdf/rr94-5.pdf from
    "Table 2.1. Comparison of String Comparators Using Last Names, First Names, and Street Names"

    >>> winkler_examples = [('SHACKLEFORD', 'SHACKELFORD'), ('DUNNINGHAM', 'CUNNIGHAM'),
    ... ('NICHLESON', 'NICHULSON'), ('JONES', 'JOHNSON'), ('MASSEY', 'MASSIE'),
    ... ('ABROMS', 'ABRAMS'), ('HARDIN', 'MARTINEZ'), ('ITMAN', 'SMITH'),
    ... ('JERALDINE', 'GERALDINE'), ('MARHTA', 'MARTHA'), ('MICHELLE', 'MICHAEL'),
    ... ('JULIES', 'JULIUS'), ('TANYA', 'TONYA'), ('DWAYNE', 'DUANE'), ('SEAN', 'SUSAN'),
    ... ('JON', 'JOHN'), ('JON', 'JAN'), ('BROOKHAVEN', 'BRROKHAVEN'),
    ... ('BROOK HALLOW', 'BROOK HLLW'), ('DECATUR', 'DECATIR'), ('FITZRUREITER', 'FITZENREITER'),
    ... ('HIGBEE', 'HIGHEE'), ('HIGBEE', 'HIGVEE'), ('LACURA', 'LOCURA'), ('IOWA', 'IONA'), ('1ST', 'IST')]

    >>> jaro_scores =   [0.970, 0.896, 0.926, 0.790, 0.889, 0.889, 0.722, 0.467, 0.926,
    ... 0.944, 0.869, 0.889, 0.867, 0.822, 0.783, 0.917, 0.000, 0.933, 0.944, 0.905,
    ... 0.856, 0.889, 0.889, 0.889, 0.833, 0.000]

    >>> winkler_scores = [0.982, 0.896, 0.956, 0.832, 0.944, 0.922, 0.722, 0.467, 0.926,
    ... 0.961, 0.921, 0.933, 0.880, 0.858, 0.805, 0.933, 0.000, 0.947, 0.967, 0.943,
    ... 0.913, 0.922, 0.922, 0.900, 0.867, 0.000]

    One way to match the values on the Winkler's paper is to provide a different
    p scaling factor for different pairs of strings, e.g.

    >>> p_factors = [0.1, 0.1, 0.1, 0.1, 0.125, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.20,
    ... 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


    >>> for (s1, s2), jscore, wscore, p in zip(winkler_examples, jaro_scores, winkler_scores, p_factors):
    ...     if (s1, s2) in [('JON', 'JAN'), ('1ST', 'IST')]:
    ...         continue  # Skip bad examples from the paper.
    ...     assert round(jaro_similarity(s1, s2), 3) == jscore
    ...     assert round(jaro_winkler_similarity(s1, s2, p=p), 3) == wscore



    This test-case proves that the output of Jaro-Winkler similarity depends on
    the product  l * p and not on the product max_l * p. Here the product max_l * p > 1
    however the product l * p <= 1

    >>> round(jaro_winkler_similarity('TANYA', 'TONYA', p=0.1, max_l=100), 3)
    0.88
    """
    if not 0 <= max_l * p <= 1:
        warnings.warn(str('The product  `max_l * p` might not fall between [0,1].Jaro-Winkler similarity might not be between 0 and 1.'))
    jaro_sim = jaro_similarity(s1, s2)
    l = 0
    for s1_i, s2_i in zip(s1, s2):
        if s1_i == s2_i:
            l += 1
        else:
            break
        if l == max_l:
            break
    return jaro_sim + l * p * (1 - jaro_sim)