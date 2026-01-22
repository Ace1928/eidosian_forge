import code
import sys
gypsh output module

gypsh is a GYP shell.  It's not really a generator per se.  All it does is
fire up an interactive Python session with a few local variables set to the
variables passed to the generator.  Like gypd, it's intended as a debugging
aid, to facilitate the exploration of .gyp structures after being processed
by the input module.

The expected usage is "gyp -f gypsh -D OS=desired_os".
