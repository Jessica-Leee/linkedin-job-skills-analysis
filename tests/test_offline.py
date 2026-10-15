"""Configuration and CSV checks: no browser, credentials or network required."""

import contextlib
import csv
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.analysis.config import DEFAULT_INPUT, parse_analysis_args
from src.scrapers.main import build_parser, main, save_dict_list_to_csv

ROOT = Path(__file__).resolve().parents[1]


class OfflineTests(unittest.TestCase):
    def test_import_and_help_do_not_load_browser_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import src.scrapers.main; "
             "assert not any(n == 'selenium' or n.startswith('selenium.') for n in sys.modules)"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            [sys.executable, "-m", "src.scrapers.main", "--help"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--query", result.stdout)

    def test_missing_auth_stops_before_any_browser_import(self):
        original_import = __import__
        def guarded_import(name, *args, **kwargs):
            if name.startswith(("selenium", "webdriver_manager")):
                self.fail("Browser dependency imported before configuration validation")
            return original_import(name, *args, **kwargs)
        with patch.dict(os.environ, {}, clear=True), patch("builtins.__import__", side_effect=guarded_import):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["--query", "synthetic engineer"])
        self.assertEqual(error.exception.code, 2)

    def test_invalid_limits_rejected(self):
        for arguments in [["--max-pages", "0"], ["--delay", "-1"], ["--delay", "nan"]]:
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    build_parser().parse_args(["--query", "example"] + arguments)

    def test_empty_query_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--query", "   "])

    def test_csv_round_trip_handles_commas_and_unicode(self):
        rows = [{"job_title": "Design, verification", "location": "Synthetic City Ω"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(save_dict_list_to_csv(rows, path))
            with path.open(newline="", encoding="utf-8") as source:
                self.assertEqual(list(csv.DictReader(source)), rows)

    def test_empty_collection_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(save_dict_list_to_csv([], path))
            self.assertFalse(path.exists())

    def test_default_input_does_not_depend_on_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                args = parse_analysis_args([])
            finally:
                os.chdir(previous)
        self.assertEqual(args.input, DEFAULT_INPUT)
        self.assertTrue(args.input.is_file())
        with args.input.open(newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 60)
        self.assertTrue(all(row["company"].startswith("Synthetic Company") for row in rows))

    def test_analysis_paths_and_clean_only_are_configurable(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "not-created"
            args = parse_analysis_args(["--input", str(DEFAULT_INPUT), "--output-dir", str(target), "--clean-only"])
            self.assertEqual(args.output_dir, target)
            self.assertTrue(args.clean_only)
            self.assertFalse(target.exists())

    def test_missing_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_analysis_args(["--input", str(Path(directory) / "missing.csv")])


if __name__ == "__main__":
    unittest.main()
