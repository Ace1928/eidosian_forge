from IPython.core import inputtransformer2 as ipt2
def test_ipython_prompt():
    for sample, expected in [IPYTHON_PROMPT, IPYTHON_PROMPT_L2, IPYTHON_PROMPT_VI_INS, IPYTHON_PROMPT_VI_NAV]:
        assert ipt2.ipython_prompt(sample.splitlines(keepends=True)) == expected.splitlines(keepends=True)