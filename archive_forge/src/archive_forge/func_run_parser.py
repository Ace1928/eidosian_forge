import os
from glob import glob
from nipype.utils.filemanip import split_filename
from textwrap import dedent
from argparse import RawDescriptionHelpFormatter
def run_parser(subparsers):
    help = dedent('\n    Process subject-level data in fitz.\n\n    This script controls the workflows that process data. Depending on the\n    workflow, the data may start from raw DICOM or Nifti files and finish\n    as processed models or group-level stats. See the documentation for the\n    pipeline (collection of workflows) for information on exactly what the\n    image inputs and outputs will be.\n\n    All fitz workflows use Nipype, so there are several common options for\n    efficiently running multiple subjects at once with differnt nipype plugins.\n    The script can thus be run for several subjects at once, and (with a large\n    enough cluster) all of the subjects can be processed in the time it takes\n    to process a single run of data linearly.\n\n    Nipype creates a cache directory to save processing time when steps are\n    re-run. If you do not delete your cache directory after running (which is\n    configured in the project file), repeated use of this script will only\n    rerun the nodes that have changes to their inputs. Otherwise, you will\n    have to rerun at the level of the workflows.\n\n\n    Examples\n    --------\n\n\n    Note that the parameter switches match any unique short version\n    of the full parameter name.\n\n    fitz run -w xnatconvert preproc onset model\n\n        Run every stage of the standar SPM fMRI pipeline for the default\n        experiment for each subject defined in $FITZ_DIR/subjects.txt.\n        The processing will be distributed locally with the MultiProc plugin\n        using 4 processes.\n\n    fitz run -s subj1 subj2 subj3 -w preproc\n\n        Run preprocessing of the default experiment for subjects `subj1`,\n        `subj2`, and `subj3`.\n\n    fitz run -s pilot_subjects -w model -e nback -n 8\n\n        Run the model workflow on the subjects enumerated in\n        $FITZ_DIR/pilot_subjects.txt with the experiment details in\n        $FITZ_DIR/nback.py. Distribute the execution locally with 8 parallel\n        processes.\n\n    Usage Details\n    -------------\n\n    ')
    if 'FITZ_DIR' in os.environ.keys():
        wf_files = glob(os.path.join(os.environ['FITZ_DIR'], '*/workflows/*.py'))
        workflows = [split_filename(wf)[1] for wf in wf_files]
    else:
        workflows = []
    parser = subparsers.add_parser('run', help='run')
    parser.description = help
    parser.formatter_class = RawDescriptionHelpFormatter
    parser.add_argument('--experiment', '-e', help='experimental paradigm')
    parser.add_argument('--model', '-m', help='model to fit')
    parser.add_argument('--workflows', '-w', nargs='*', choices=workflows, help='which workflows to run')
    parser.add_argument('--subjects', '-s', nargs='*', dest='subjects', help='list of subject ids, name of file in lyman directory, or full path to text file with subject ids')
    parser.add_argument('--plugin', '-p', default='multiproc', choices=['linear', 'multiproc', 'ipython', 'torque', 'sge', 'slurm'], help='worklow execution plugin')
    parser.add_argument('--nprocs', '-n', default=4, type=int, help='number of MultiProc processes to use')
    parser.add_argument('--queue', '-q', help='which queue for scheduler execution')
    parser.add_argument('--dontrun', action='store_true', help="don't actually execute the workflows")
    return parser