"""Standard-library-only input configuration for the analysis command."""

import argparse
from pathlib import Path

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "examples" / "synthetic_jobs.csv"


def parse_analysis_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze a job-posting CSV")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--clean-only", action="store_true",
                        help="Write a cleaned CSV without NLP reports or NLTK downloads")
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"Input CSV not found: {args.input}")
    return args
