"""Deterministic vectors and public API contracts for the raw BDSP RNG."""

import unittest

from core.bdsp.rng import Xorshift128


class Xorshift128Tests(unittest.TestCase):
    INITIAL = (1, 2, 3, 4)
    # Fixed regression vectors for the raw BDSP Xorshift128 transition,
    # independently derived and cross-checked against established BDSP RNG
    # implementations/research. Data only; no third-party implementation code copied.
    VECTORS = (
        (
            INITIAL,
            (
                0x0000080D, 0x0000181F, 0x00000004, 0x00002020,
                0x0040004D, 0x008020BA, 0x0080008E, 0x0180219E,
            ),
            (0x10C3C7BC, 0xE9F3390F, 0xF16883E2, 0x50AEC61F),
        ),
        (
            (0x12345678, 0x9ABCDEF0, 0x0FEDCBA9, 0x87654321),
            (
                0x37524223, 0x4B655167, 0x2AB46B21, 0x8765591D,
                0x22835088, 0x430CD3D6, 0xCA6855F5, 0x67685014,
            ),
            (0xEBCFA769, 0xA4B8826B, 0x0D28A6C5, 0x4788B91F),
        ),
    )

    def test_first_eight_outputs_and_state(self):
        for initial, outputs, _ in self.VECTORS:
            with self.subTest(initial=initial):
                rng = Xorshift128(initial)
                self.assertEqual(tuple(rng.next_u32() for _ in range(8)), outputs)
                self.assertEqual(rng.get_state(), outputs[-4:])

    def test_state_after_exactly_1000_calls(self):
        for initial, _, expected in self.VECTORS:
            with self.subTest(initial=initial):
                rng = Xorshift128(initial)
                rng.advance(1000)
                self.assertEqual(rng.get_state(), expected)

    def test_boundary_single_transitions(self):
        cases = (
            ((0x80000000, 0, 0, 0), 0x80800000),
            ((0, 0, 0, 0x80000000), 0x80001000),
            ((0xFFFFFFFF,) * 4, 0xFFFFE7F8),
        )
        for initial, expected in cases:
            with self.subTest(initial=initial):
                rng = Xorshift128(initial)
                result = rng.next_u32()
                self.assertIs(type(result), int)
                self.assertEqual(result, expected)
                self.assertEqual(rng.get_state(), (*initial[1:], expected))

    def test_constructor_consumes_no_calls(self):
        self.assertEqual(Xorshift128(self.INITIAL).get_state(), self.INITIAL)

    def test_constructor_rejects_wrong_word_count(self):
        for length in (0, 1, 2, 3, 5, 8):
            with self.subTest(length=length):
                with self.assertRaises(ValueError):
                    Xorshift128([0] * length)

    def test_constructor_rejects_invalid_word_types(self):
        for value in (True, False, 1.0, "1", None, complex(1, 0)):
            for index in range(4):
                with self.subTest(value=value, index=index):
                    words = list(self.INITIAL)
                    words[index] = value
                    with self.assertRaises(TypeError):
                        Xorshift128(words)

    def test_constructor_rejects_out_of_range_words(self):
        for value in (-1, 0x100000000, 1 << 128):
            for index in range(4):
                with self.subTest(value=value, index=index):
                    words = list(self.INITIAL)
                    words[index] = value
                    with self.assertRaises(ValueError):
                        Xorshift128(words)

    def test_constructor_copies_supplied_sequence(self):
        words = list(self.INITIAL)
        rng = Xorshift128(words)
        words[0] = 999
        words.append(5)
        self.assertEqual(rng.get_state(), self.INITIAL)
        self.assertEqual(rng.next_u32(), 0x0000080D)
        self.assertEqual(words, [999, 2, 3, 4, 5])

    def test_all_zero_state(self):
        rng = Xorshift128((0, 0, 0, 0))
        self.assertEqual([rng.next_u32() for _ in range(8)], [0] * 8)
        rng.advance(1000)
        self.assertEqual(rng.get_state(), (0, 0, 0, 0))

    def test_advance_zero(self):
        rng = Xorshift128(self.INITIAL)
        self.assertIsNone(rng.advance(0))
        self.assertEqual(rng.get_state(), self.INITIAL)

    def test_advance_equivalence(self):
        for steps in (1, 2, 8, 31, 1000):
            with self.subTest(steps=steps):
                advanced = Xorshift128(self.INITIAL)
                repeated = Xorshift128(self.INITIAL)
                self.assertIsNone(advanced.advance(steps))
                for _ in range(steps):
                    repeated.next_u32()
                self.assertEqual(advanced.get_state(), repeated.get_state())
                self.assertEqual(advanced.next_u32(), repeated.next_u32())

    def test_advance_rejects_negative_count(self):
        rng = Xorshift128(self.INITIAL)
        with self.assertRaises(ValueError):
            rng.advance(-1)
        self.assertEqual(rng.get_state(), self.INITIAL)

    def test_advance_rejects_invalid_types(self):
        rng = Xorshift128(self.INITIAL)
        for steps in (True, False, 0.0, 1.5, "1", None):
            with self.subTest(steps=steps):
                with self.assertRaises(TypeError):
                    rng.advance(steps)
                self.assertEqual(rng.get_state(), self.INITIAL)

    def test_get_state_is_immutable_and_does_not_advance(self):
        rng = Xorshift128(self.INITIAL)
        snapshot = rng.get_state()
        self.assertIs(type(snapshot), tuple)
        self.assertEqual(rng.get_state(), self.INITIAL)
        with self.assertRaises(TypeError):
            snapshot[0] = 0
        self.assertEqual(rng.next_u32(), 0x0000080D)
        self.assertEqual(snapshot, self.INITIAL)

    def test_clone_independence(self):
        original = Xorshift128(self.INITIAL)
        original.advance(3)
        snapshot = original.get_state()
        cloned = original.clone()
        self.assertIsNot(cloned, original)
        self.assertEqual(cloned.get_state(), snapshot)
        self.assertEqual(original.get_state(), snapshot)
        self.assertEqual(original.next_u32(), 0x00002020)
        self.assertEqual(cloned.get_state(), snapshot)
        self.assertEqual(cloned.next_u32(), 0x00002020)
        original_state = original.get_state()
        cloned.advance(5)
        self.assertEqual(original.get_state(), original_state)
        self.assertNotEqual(cloned.get_state(), original_state)


if __name__ == "__main__":
    unittest.main()
