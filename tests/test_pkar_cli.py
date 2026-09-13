"""CLI parsing, output, and integration with the existing BDSP search."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from core.bdsp.ids import BDSPIDResult
from pkar_cli import main


class PKARCLITests(unittest.TestCase):
    def run_cli(self, *args):
        output, errors = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            try:
                code = main(args)
            except SystemExit as error:
                code = error.code
        return code, output.getvalue(), errors.getvalue()

    def arguments(self, state=None, target="999999", start=None, stop="1"):
        args = ["--state", *(state or ("82605DF2", "00000000", "00000000", "00000000")),
                "--target", target, "--stop", stop]
        if start is not None:
            args.extend(("--start", start))
        return args

    def assert_invalid(self, args):
        code, output, errors = self.run_cli(*args)
        self.assertNotEqual(code, 0)
        self.assertEqual(output, "")
        self.assertIn("error:", errors)
        self.assertNotIn("Traceback", errors)

    def test_999999_complete_output_and_default_start(self):
        code, output, errors = self.run_cli(*self.arguments())
        self.assertEqual(code, 0)
        self.assertEqual(errors, "")
        self.assertEqual(output, (
            "Advance: 0\n"
            "Display ID: 999999\n"
            "TID16: 16959\n"
            "SID16: 15\n"
            "TSV: 1059\n"
            "Combined: 000F423F\n"
            "Raw outputs: 800F423F\n"
            "Calls consumed: 1\n"
            "State after: 00000000 00000000 00000000 800F423F\n"
        ))

    def test_zero_display_id(self):
        code, output, errors = self.run_cli(*self.arguments(
            state=("9DA3A58D", "0", "0", "0"), target="000000"
        ))
        self.assertEqual((code, errors), (0, ""))
        self.assertIn("Display ID: 000000\n", output)
        self.assertIn("Combined: 000F4240\n", output)

    def test_leading_zero_target_and_rejected_draw(self):
        code, output, errors = self.run_cli(*self.arguments(
            state=("A4848080", "0", "0", "0"), target="004096"
        ))
        self.assertEqual((code, errors), (0, ""))
        self.assertIn("Display ID: 004096\n", output)
        self.assertIn("Raw outputs: 80000000 80001000\n", output)
        self.assertIn("Calls consumed: 2\n", output)
        self.assertIn("State after: 00000000 00000000 80000000 80001000\n", output)

    def test_no_match(self):
        code, output, errors = self.run_cli(*self.arguments(
            state=("1", "2", "3", "4"), stop="3"
        ))
        self.assertEqual((code, errors), (0, ""))
        self.assertEqual(output, "No matching Trainer ID found in advances [0, 3).\n")

    def test_multiple_matches(self):
        code, output, errors = self.run_cli(*self.arguments(
            state=("A5A43414", "04809010", "04809010", "04809010"),
            target="876352", stop="5",
        ))
        self.assertEqual((code, errors), (0, ""))
        blocks = output.strip().split("\n\n")
        self.assertEqual(len(blocks), 5)
        for advance, block in enumerate(blocks):
            self.assertTrue(block.startswith(f"Advance: {advance}\n"))
            self.assertIn("Display ID: 876352\n", block)
            self.assertIn(f"Calls consumed: {5 - advance}\n", block)

    def test_hex_prefix_and_lowercase(self):
        expected = self.run_cli(*self.arguments())
        for state in (
            ("0x82605DF2", "0x0", "0x0", "0x0"),
            ("82605df2", "0", "0", "0"),
            ("0X82605df2", "0", "0x0", "00000000"),
        ):
            with self.subTest(state=state):
                self.assertEqual(self.run_cli(*self.arguments(state=state)), expected)

    def test_decimal_looking_state_is_hex_and_search_is_streamed(self):
        output = StringIO()
        first = BDSPIDResult(0, 0xF423F, (0x800F423F,), (0, 0, 0, 0x800F423F))

        def matches():
            yield first
            self.assertIn("Display ID: 999999", output.getvalue())
            yield first

        with patch("pkar_cli.search_ids", return_value=matches()) as search:
            with redirect_stdout(output):
                code = main(self.arguments(state=("10", "20", "30", "40"), stop="5000000"))
            search.assert_called_once_with([16, 32, 48, 64], 999999, start=0, stop=5000000)
        self.assertEqual(code, 0)
        self.assertEqual(output.getvalue().count("Display ID: 999999"), 2)

    def test_invalid_hex(self):
        for word in ("G123", "0x", "1_000", "-1", "0xGG", ""):
            with self.subTest(word=word):
                self.assert_invalid(self.arguments(state=(word, "0", "0", "0")))

    def test_state_out_of_range(self):
        for word in ("100000000", "0x100000000"):
            with self.subTest(word=word):
                self.assert_invalid(self.arguments(state=(word, "0", "0", "0")))

    def test_invalid_targets(self):
        for target in ("-1", "0000000", "letters", "1000000", "1.0", "+1", "", "１２３"):
            with self.subTest(target=target):
                self.assert_invalid(self.arguments(target=target))

    def test_short_decimal_target(self):
        code, output, errors = self.run_cli(*self.arguments(
            state=("9DA3A58D", "0", "0", "0"), target="0"
        ))
        self.assertEqual((code, errors), (0, ""))
        self.assertIn("Display ID: 000000\n", output)

    def test_invalid_advances(self):
        for field in ("start", "stop"):
            for value in ("-1", "abc", "1.5", "0x10", "+1", "1_000", ""):
                with self.subTest(field=field, value=value):
                    self.assert_invalid(self.arguments(**{field: value}))

    def test_stop_before_start(self):
        self.assert_invalid(self.arguments(start="2", stop="1"))

    def test_empty_interval(self):
        code, output, errors = self.run_cli(*self.arguments(start="3", stop="3"))
        self.assertEqual((code, errors), (0, ""))
        self.assertEqual(output, "No matching Trainer ID found in advances [3, 3).\n")

    def test_explicit_decimal_interval(self):
        code, output, errors = self.run_cli(*self.arguments(
            state=("1", "2", "3", "4"), target="489823", start="01", stop="02"
        ))
        self.assertEqual((code, errors), (0, ""))
        self.assertIn("Advance: 1\n", output)

    def test_missing_required_arguments_and_state_count(self):
        for args in ([], ["--state", "1", "2", "3", "--target", "0", "--stop", "1"],
                     ["--state", "1", "2", "3", "4", "--target", "0"],
                     ["--state", "1", "2", "3", "4", "--stop", "1"]):
            with self.subTest(args=args):
                self.assert_invalid(args)

    def test_script_exit_codes(self):
        script = Path(__file__).resolve().parents[1] / "pkar_cli.py"
        for target, expected_code in (("999999", 0), ("123456", 0), ("invalid", 2)):
            with self.subTest(target=target):
                result = subprocess.run(
                    [sys.executable, "-B", str(script), *self.arguments(target=target)],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, expected_code)
                self.assertNotIn("Traceback", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
