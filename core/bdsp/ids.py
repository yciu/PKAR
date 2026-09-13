"""BDSP Trainer ID generation from candidate offsets in the raw RNG stream."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from core.bdsp.rng import Xorshift128


MASK = 0xFFFFFFFF


@dataclass(frozen=True)
class BDSPIDResult:
    """One candidate, including retry draws and the state after acceptance."""

    advance: int
    combined: int
    raw_outputs: tuple[int, ...]
    state_after: tuple[int, int, int, int]

    @property
    def tid16(self) -> int:
        return self.combined & 0xFFFF

    @property
    def sid16(self) -> int:
        return self.combined >> 16

    @property
    def display_id(self) -> int:
        """Numeric display ID; format with six decimal digits when displaying."""
        return self.combined % 1_000_000

    @property
    def calls_consumed(self) -> int:
        return len(self.raw_outputs)

    @property
    def tsv(self) -> int:
        return (self.tid16 ^ self.sid16) >> 4


def _validate_advance(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not a boolean")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _generate_candidate(rng: Xorshift128, advance: int) -> BDSPIDResult:
    raw_outputs = []
    while True:
        raw = rng.next_u32()
        raw_outputs.append(raw)
        combined = ((raw % MASK) + 0x80000000) & MASK
        if combined != 0:
            return BDSPIDResult(
                advance, combined, tuple(raw_outputs), rng.get_state()
            )


def generate_id(state: Sequence[int], *, advance: int = 0) -> BDSPIDResult:
    """Skip advance raw calls privately, then generate until a nonzero ID."""
    _validate_advance(advance, "advance")
    rng = Xorshift128(state)
    rng.advance(advance)
    return _generate_candidate(rng, advance)


def iter_ids(
    state: Sequence[int], *, start: int = 0, stop: int
) -> Iterator[BDSPIDResult]:
    """Evaluate independent starts in [start, stop), retaining duplicate IDs.

    Snapshot and validate inputs immediately. Retry draws may extend past stop;
    each subsequent candidate still starts exactly one raw call later.
    """
    _validate_advance(start, "start")
    _validate_advance(stop, "stop")
    if stop < start:
        raise ValueError("stop must be greater than or equal to start")
    base = Xorshift128(state)

    def candidates() -> Iterator[BDSPIDResult]:
        if start == stop:
            return
        base.advance(start)
        for advance in range(start, stop):
            yield _generate_candidate(base.clone(), advance)
            base.next_u32()

    return candidates()


def search_ids(
    state: Sequence[int], target: int, *, start: int = 0, stop: int
) -> Iterator[BDSPIDResult]:
    """Lazily match numeric displayed IDs in [start, stop), keeping duplicates.

    Validate and snapshot inputs immediately. Leading zeros are presentation
    only: format display_id with six decimal digits when displaying.
    """
    if isinstance(target, bool) or not isinstance(target, int):
        raise TypeError("target must be an integer, not a boolean")
    if not 0 <= target <= 999999:
        raise ValueError("target must be between 0 and 999999 inclusive")
    candidates = iter_ids(state, start=start, stop=stop)
    return (result for result in candidates if result.display_id == target)
