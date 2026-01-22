
The Model Zoo.

This file contains a list of all the models in the model zoo, the path to
load them, agents & tasks associated (e.g. they were trained using) and a
description. Using the path you should be able to download and use the model
automatically, e.g.:

... code-block:

   parlai interactive --model-file
       "zoo:wikipedia_20161221/tfidf_retriever/drqa_docs"


There are a number of guidelines you should follow in the zoo:

- You should choose the best directory name as possible. An input of
  ``zoo:PROJECTNAME/MODELNAME/FILENAME`` will attempt to use a build script from
  parlai/zoo/PROJECTNAME/MODELNAME.py.
- You should include all of the following fields:
    * title: the name of the entry:
    * id: corresponds to PROJECTNAME
    * description: describe the entry in reasonable detail. It should be at least
      a couple sentences.
    * example: an example command to chat with or evaluate the model
    * result: the expected output from running the model. You are strongly encouraged
      to make a nightly test which verifies this result.
    * external_website: if applicable, an external website related to the zoo to
      link to.
    * project: if applicable, a link to the project folder. You must have either
      external_website or project.
    * example2 and result2 (optional): additional examples to run.

- As much as possible, you should try to include two examples: one to generate
  some key metrics (e.g. from a paper) and one to actually chat with the model
  using interactive.py. Both should strongly attempt to minimize mandatory
  command line flags.
