"""Raw BDSP Xorshift128 transitions over four unsigned 32-bit words."""

from collections.abc import Sequence


MASK = 0xFFFFFFFF


class Xorshift128:
    """Mutable RNG with state ordered as (x, y, z, w), before the next call."""

    def __init__(self, state: Sequence[int]) -> None:
        words = tuple(state)
        if len(words) != 4:
            raise ValueError("state must contain exactly four words")
        for word in words:
            if isinstance(word, bool) or not isinstance(word, int):
                raise TypeError("state words must be integers, not booleans")
            if not 0 <= word <= MASK:
                raise ValueError("state words must be unsigned 32-bit values")
        self._state = words

    def next_u32(self) -> int:
        """Advance once and return the new fourth word, without conversion."""
        x, y, z, w = self._state
        t = (x ^ ((x << 11) & MASK)) & MASK
        t = (t ^ (t >> 8)) & MASK
        new_w = (w ^ (w >> 19) ^ t) & MASK
        self._state = (y, z, w, new_w)
        return new_w

    def advance(self, steps: int) -> None:
        """Discard exactly steps outputs; zero leaves the state unchanged."""
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("steps must be an integer, not a boolean")
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        for _ in range(steps):
            self.next_u32()

    def get_state(self) -> tuple[int, int, int, int]:
        """Return an immutable snapshot without advancing."""
        return self._state

    def clone(self) -> "Xorshift128":
        """Return an independent generator at the same state."""
        return Xorshift128(self._state)
