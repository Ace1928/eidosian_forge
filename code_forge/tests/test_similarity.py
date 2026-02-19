from code_forge.library.similarity import (
    build_fingerprint,
    hamming_distance64,
    normalize_code_text,
    normalized_hash,
    simhash64,
    split_identifier,
    token_jaccard,
    tokenize_code_text,
)


def test_identifier_split_and_tokenize() -> None:
    assert split_identifier("parseHTTPResponse") == ["parse", "httpresponse"]
    tokens = tokenize_code_text("def parse_http_response(statusCode): return statusCode")
    assert "parse" in tokens
    assert "http" in tokenize_code_text("parse_http_response")


def test_normalization_is_comment_and_literal_stable() -> None:
    left = "x = 1  # comment\nname = 'alice'\n"
    right = "x=99\nname = \"bob\" // another comment\n"

    assert normalize_code_text(left) == normalize_code_text(right)
    assert normalized_hash(left) == normalized_hash(right)


def test_simhash_and_jaccard_similarity() -> None:
    a = tokenize_code_text("def total_price(items): return sum(items)")
    b = tokenize_code_text("def compute_total_price(items): return sum(items)")
    c = tokenize_code_text("def render_html(page): return page")

    sim_a = simhash64(a)
    sim_b = simhash64(b)
    sim_c = simhash64(c)

    assert hamming_distance64(sim_a, sim_b) < hamming_distance64(sim_a, sim_c)
    assert token_jaccard(a, b) > token_jaccard(a, c)


def test_build_fingerprint() -> None:
    norm, sim, token_count = build_fingerprint("def add(a, b):\n    return a + b\n")
    assert len(norm) == 64
    assert isinstance(sim, int)
    assert token_count > 0
