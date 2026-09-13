"""Command-line search for a displayed BDSP Trainer ID."""

import argparse
from collections.abc import Sequence
import re

from core.bdsp.ids import BDSPIDResult, search_ids


def _state_word(text: str) -> int:
    if re.fullmatch(r"(?:0[xX])?[0-9a-fA-F]+", text) is None:
        raise argparse.ArgumentTypeError("state words must be hexadecimal")
    return int(text, 16)


def _target(text: str) -> int:
    if re.fullmatch(r"[0-9]{1,6}", text) is None:
        raise argparse.ArgumentTypeError("target must contain 1 to 6 decimal digits")
    return int(text, 10)


def _advance(text: str) -> int:
    if re.fullmatch(r"[0-9]+", text) is None:
        raise argparse.ArgumentTypeError("advances must be nonnegative decimal integers")
    try:
        return int(text, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("advance value is too long to parse") from error


def _format_result(result: BDSPIDResult) -> str:
    return "\n".join((
        f"Advance: {result.advance}",
        f"Display ID: {result.display_id:06d}",
        f"TID16: {result.tid16}",
        f"SID16: {result.sid16}",
        f"TSV: {result.tsv}",
        f"Combined: {result.combined:08X}",
        "Raw outputs: " + " ".join(f"{raw:08X}" for raw in result.raw_outputs),
        f"Calls consumed: {result.calls_consumed}",
        "State after: " + " ".join(f"{word:08X}" for word in result.state_after),
    ))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state", nargs=4, required=True, type=_state_word,
        metavar=("S0", "S1", "S2", "S3"),
        help="four hexadecimal state words, with optional 0x prefixes",
    )
    parser.add_argument(
        "--target", required=True, type=_target,
        help="displayed Trainer ID, 000000 through 999999",
    )
    parser.add_argument("--start", type=_advance, default=0, help="inclusive starting advance (default: 0)")
    parser.add_argument("--stop", type=_advance, required=True, help="exclusive stopping advance")
    args = parser.parse_args(argv)
    try:
        matches = search_ids(args.state, args.target, start=args.start, stop=args.stop)
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    found = False
    for result in matches:
        if found:
            print()
        print(_format_result(result))
        found = True
    if not found:
        print(f"No matching Trainer ID found in advances [{args.start}, {args.stop}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
