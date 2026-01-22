from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Protocol, Tuple, Union
import interegular
from lark import Lark
from outlines import grammars
from outlines.caching import cache
from outlines.fsm.regex import create_fsm_index_tokenizer, make_deterministic_fsm
def get_next_instruction(self, state: int) -> Instruction:
    """Generate an instruction for the next step.

        Upon initialization, the CFG incremental parser is used to determine the
        first regex and construct the first FSM to generate the first terminal.

        This FSM is used for proposals until either:

        - The FSM is exhausted, and its only remaining option is the EOS token,
          in which case we feed the generated terminal to the
          CFG incremental parser and allow it to propose the next regex
          corresponding to the next set of valid terminals.
        - The current FSM can be exhausted, but the EOS token is not the only
          remaining option. In this case we allow proposal of current terminal
          extensions, store the current FSM and its state, then also use the CFG
          parser to propose a new regex corresponding to terminating the current
          terminal and starting the next one. The model can then sample from
          either of these sets to determine whether to extend the current
          terminal or terminate it and start the next one.

        The CFG incremental parser is allowed to propose the EOS token from any accepting state,
        and once it is generated, the FSM will continue to always generate the EOS token.

        Parameters
        ----------
        state
            The current state of the FSM.

        Returns
        -------
        A list that contains the tokens to mask.

        """
    if self.is_final_state(state):
        return Write([self.tokenizer.eos_token_id])
    proposal: List[int] = []
    if self.generation != '':
        if self.check_last:
            proposer = self.regex_fsm_last
        else:
            proposer = self.regex_fsm
        instruction = proposer.get_next_instruction(state)
        if isinstance(instruction, Write):
            proposal += instruction.tokens
        else:
            proposal += instruction.tokens
        if self.tokenizer.eos_token_id not in proposal:
            return Generate(proposal)
        self.check_last = False
        proposal = [x for x in proposal if x != self.tokenizer.eos_token_id]
        if len(proposal) > 0:
            self.check_last = True
            self.proposal_last = proposal.copy()
            self.regex_fsm_last = proposer
    interactive = self.parser.parse_interactive(self.generation)
    interactive.exhaust_lexer()
    options = {self.terminal_regexps[x] for x in interactive.accepts()}
    options |= {self.terminal_regexps[x] for x in self.parser.lexer_conf.ignore}
    if self.terminal_regexps['$END'] in options:
        options.remove(self.terminal_regexps['$END'])
        if len(options) == 0:
            return Write([self.tokenizer.eos_token_id])
        self.allow_eos = True
        options.add('')
        assert len(options) > 1
    regex_string = '(' + '|'.join(['(' + x + ')' for x in options]) + ')'
    self.regex_fsm = RegexGuide(regex_string, self.tokenizer)
    self.reset_state = True
    instruction = self.regex_fsm.get_next_instruction(self.start_state)
    if isinstance(instruction, Write):
        proposal += instruction.tokens
    else:
        proposal += instruction.tokens
    if self.allow_eos:
        self.allow_eos = False
    else:
        proposal = [x for x in proposal if x != self.tokenizer.eos_token_id]
        assert len(proposal) > 0
    return Generate(proposal)