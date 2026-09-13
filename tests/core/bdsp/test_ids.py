"""Fixed, independently derived BDSP ID regression data; no external code copied."""

from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import Mock, patch

from core.bdsp.ids import BDSPIDResult, generate_id, iter_ids, search_ids


class BDSPIDTests(unittest.TestCase):
    A = (1, 2, 3, 4)
    B = (0x12345678, 0x9ABCDEF0, 0x0FEDCBA9, 0x87654321)
    C = (0xFFFFFFFF,) * 4
    D = (0xA4848080, 0, 0, 0)
    E = (0xA4848080, 0x04809010, 0, 0)
    F = (0xA5A43414, 0x04809010, 0x04809010, 0x04809010)
    F_END = (0x80000000, 0x80000000, 0x80000000, 0x00801000)

    # Approved research vectors, cross-checked against established BDSP tools.
    # initial, advance, raw, combined, TID16, SID16, display ID, TSV, ending state
    NORMAL = (
        (A, 0, 0x0000080D, 0x8000080D, 2061, 32768, 485709, 2176,
         (2, 3, 4, 0x0000080D)),
        (A, 1, 0x0000181F, 0x8000181F, 6175, 32768, 489823, 2433,
         (3, 4, 0x0000080D, 0x0000181F)),
        (A, 2, 0x00000004, 0x80000004, 4, 32768, 483652, 2048,
         (4, 0x0000080D, 0x0000181F, 4)),
        (A, 999, 0x50AEC61F, 0xD0AEC61F, 50719, 53422, 114911, 363,
         (0x10C3C7BC, 0xE9F3390F, 0xF16883E2, 0x50AEC61F)),
        (B, 0, 0x37524223, 0xB7524223, 16931, 46930, 621411, 3927,
         (0x9ABCDEF0, 0x0FEDCBA9, 0x87654321, 0x37524223)),
        (B, 1, 0x4B655167, 0xCB655167, 20839, 52069, 414823, 2464,
         (0x0FEDCBA9, 0x87654321, 0x37524223, 0x4B655167)),
        (B, 2, 0x2AB46B21, 0xAAB46B21, 27425, 43700, 950625, 3097,
         (0x87654321, 0x37524223, 0x4B655167, 0x2AB46B21)),
        (B, 999, 0x4788B91F, 0xC788B91F, 47391, 51080, 626271, 2025,
         (0xEBCFA769, 0xA4B8826B, 0x0D28A6C5, 0x4788B91F)),
        (C, 0, 0xFFFFE7F8, 0x7FFFE7F8, 59384, 32767, 477496, 2432,
         (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFE7F8)),
        (C, 1, 0xFFFFFFFF, 0x80000000, 0, 32768, 483648, 2048,
         (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFE7F8, 0xFFFFFFFF)),
    )

    def test_normal_vectors(self):
        for initial, advance, raw, combined, tid, sid, display, tsv, end in self.NORMAL:
            with self.subTest(initial=initial, advance=advance):
                result = generate_id(initial, advance=advance)
                self.assertEqual(result, BDSPIDResult(advance, combined, (raw,), end))
                self.assertEqual(result.tid16, tid)
                self.assertEqual(result.sid16, sid)
                self.assertEqual(result.display_id, display)
                self.assertEqual(result.tsv, tsv)
                self.assertEqual(result.calls_consumed, 1)

    def test_maximum_combined_is_accepted(self):
        result = generate_id((0x9C7C7F80, 0, 0, 0))
        self.assertEqual(result, BDSPIDResult(
            0, 0xFFFFFFFF, (0x7FFFFFFF,), (0, 0, 0, 0x7FFFFFFF)
        ))
        self.assertEqual((result.tid16, result.sid16), (65535, 65535))
        self.assertEqual(result.display_id, 967295)
        self.assertEqual(result.tsv, 0)

    def test_zero_display_id_is_accepted(self):
        result = generate_id((0x9DA3A58D, 0, 0, 0))
        self.assertEqual(result, BDSPIDResult(
            0, 0x000F4240, (0x800F4240,), (0, 0, 0, 0x800F4240)
        ))
        self.assertEqual((result.tid16, result.sid16), (16960, 15))
        self.assertEqual(result.display_id, 0)
        self.assertEqual(f"{result.display_id:06d}", "000000")
        self.assertEqual(result.calls_consumed, 1)

    def test_display_id_reaches_999999(self):
        result = generate_id((0x82605DF2, 0, 0, 0), advance=0)
        self.assertEqual(result, BDSPIDResult(
            0, 0x000F423F, (0x800F423F,), (0, 0, 0, 0x800F423F)
        ))
        self.assertEqual(result.tid16, 16959)
        self.assertEqual(result.sid16, 15)
        self.assertEqual(result.display_id, 999999)
        self.assertEqual(result.tsv, 1059)
        self.assertEqual(result.calls_consumed, 1)
        self.assertEqual(f"{result.display_id:06d}", "999999")

    def test_all_zero_rng_state(self):
        for result in iter_ids((0, 0, 0, 0), stop=3):
            self.assertEqual(result.combined, 0x80000000)
            self.assertEqual(result.raw_outputs, (0,))
            self.assertEqual(result.state_after, (0, 0, 0, 0))
            self.assertEqual((result.tid16, result.sid16), (0, 32768))
            self.assertEqual(result.display_id, 483648)
            self.assertEqual(result.calls_consumed, 1)

    def test_single_and_double_rejection(self):
        cases = (
            (self.D, (0x80000000, 0x80001000),
             (0, 0, 0x80000000, 0x80001000)),
            (self.E, (0x80000000, 0x80000000, 0x80001000),
             (0, 0x80000000, 0x80000000, 0x80001000)),
        )
        for initial, raws, end in cases:
            with self.subTest(initial=initial):
                result = generate_id(initial)
                self.assertEqual(result, BDSPIDResult(0, 0x1000, raws, end))
                self.assertEqual((result.tid16, result.sid16), (4096, 0))
                self.assertEqual(result.display_id, 4096)
                self.assertEqual(result.tsv, 256)
                self.assertEqual(result.calls_consumed, len(raws))

    def test_four_rejections(self):
        result = generate_id(self.F)
        self.assertEqual(result, BDSPIDResult(
            0, 0x80801000, (0x80000000,) * 4 + (0x00801000,), self.F_END
        ))
        self.assertEqual(result.calls_consumed, 5)
        self.assertEqual((result.tid16, result.sid16), (4096, 32896))
        self.assertEqual(result.display_id, 876352)
        self.assertEqual(result.tsv, 2312)

    def test_candidate_windows_preserve_duplicates_and_retry_traces(self):
        results = list(iter_ids(self.F, stop=5))
        self.assertEqual([r.advance for r in results], [0, 1, 2, 3, 4])
        self.assertEqual([r.calls_consumed for r in results], [5, 4, 3, 2, 1])
        for advance, result in enumerate(results):
            with self.subTest(advance=advance):
                self.assertEqual(result.combined, 0x80801000)
                self.assertEqual(result.state_after, self.F_END)
                self.assertEqual(
                    result.raw_outputs, (0x80000000,) * (4 - advance) + (0x00801000,)
                )
                self.assertEqual(result, generate_id(self.F, advance=advance))

    def test_retries_can_extend_past_stop(self):
        results = list(iter_ids(self.F, start=2, stop=3))
        self.assertEqual(results, [BDSPIDResult(
            2, 0x80801000, (0x80000000, 0x80000000, 0x00801000), self.F_END
        )])

    def test_iteration_nonzero_start_and_exclusive_stop(self):
        results = list(iter_ids(self.A, start=1, stop=3))
        self.assertEqual([r.advance for r in results], [1, 2])
        self.assertEqual([r.combined for r in results], [0x8000181F, 0x80000004])
        self.assertEqual([r.raw_outputs for r in results], [(0x181F,), (4,)])
        self.assertEqual(
            list(iter_ids(self.B, start=999, stop=1000)),
            [generate_id(self.B, advance=999)],
        )

    def test_empty_iteration(self):
        for offset in (0, 5, 999):
            with self.subTest(offset=offset):
                self.assertEqual(list(iter_ids(self.A, start=offset, stop=offset)), [])

    def test_input_is_not_mutated_and_iteration_snapshots_input(self):
        words = list(self.A)
        generate_id(words, advance=999)
        self.assertEqual(words, list(self.A))
        self.assertEqual(len(list(iter_ids(words, stop=3))), 3)
        self.assertEqual(words, list(self.A))
        iterator = iter_ids(words, stop=1)
        words[:] = [0, 0, 0, 0]
        self.assertEqual(next(iterator), generate_id(self.A))
        self.assertEqual(words, [0, 0, 0, 0])

    def test_result_is_immutable(self):
        result = generate_id(self.A)
        for name in (
            "advance", "combined", "raw_outputs", "state_after",
            "tid16", "sid16", "display_id", "calls_consumed", "tsv",
        ):
            with self.subTest(name=name):
                with self.assertRaises(FrozenInstanceError):
                    setattr(result, name, 0)
        self.assertIsInstance(result.raw_outputs, tuple)
        self.assertIsInstance(result.state_after, tuple)
        with self.assertRaises(TypeError):
            result.raw_outputs[0] = 0
        with self.assertRaises(TypeError):
            result.state_after[0] = 0

    def test_invalid_offset_types(self):
        for value in (True, False, 1.0, "1", None):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    generate_id(self.A, advance=value)
                with self.assertRaises(TypeError):
                    iter_ids(self.A, start=value, stop=5)
                with self.assertRaises(TypeError):
                    iter_ids(self.A, stop=value)

    def test_negative_offsets(self):
        with self.assertRaises(ValueError):
            generate_id(self.A, advance=-1)
        with self.assertRaises(ValueError):
            iter_ids(self.A, start=-1, stop=5)
        with self.assertRaises(ValueError):
            iter_ids(self.A, stop=-1)

    def test_reversed_interval(self):
        with self.assertRaises(ValueError):
            iter_ids(self.A, start=2, stop=1)

    def test_state_validation_is_delegated(self):
        cases = (
            ((), ValueError),
            ((1, 2, 3), ValueError),
            ((1, 2, 3, 4, 5), ValueError),
            ((True, 2, 3, 4), TypeError),
            ((1, 2, False, 4), TypeError),
            ((1, 2, 3, 4.0), TypeError),
            ((1, "2", 3, 4), TypeError),
            ((-1, 2, 3, 4), ValueError),
            ((1, 2, 3, 0x100000000), ValueError),
            (None, TypeError),
        )
        for state, error in cases:
            with self.subTest(state=state):
                with self.assertRaises(error):
                    generate_id(state)
                with self.assertRaises(error):
                    iter_ids(state, stop=1)
                with self.assertRaises(error):
                    iter_ids(state, stop=0)


class BDSPIDSearchTests(unittest.TestCase):
    A = (1, 2, 3, 4)
    D = (0xA4848080, 0, 0, 0)
    F = (0xA5A43414, 0x04809010, 0x04809010, 0x04809010)
    F_END = (0x80000000, 0x80000000, 0x80000000, 0x00801000)

    def test_targets_and_six_digit_formatting(self):
        cases = (
            ((0x82605DF2, 0, 0, 0), 999999, "999999"),
            ((0x9DA3A58D, 0, 0, 0), 0, "000000"),
            (self.D, 4096, "004096"),
        )
        for state, target, formatted in cases:
            with self.subTest(target=target):
                results = list(search_ids(state, target, stop=1))
                self.assertEqual([r.advance for r in results], [0])
                self.assertEqual(results[0].display_id, target)
                self.assertEqual(f"{results[0].display_id:06d}", formatted)

    def test_no_match(self):
        self.assertEqual(list(search_ids(self.A, 999999, stop=3)), [])

    def test_nonzero_match(self):
        results = list(search_ids(self.A, 489823, stop=3))
        self.assertEqual(results, [BDSPIDResult(
            1, 0x8000181F, (0x0000181F,), (3, 4, 0x0000080D, 0x0000181F)
        )])

    def test_interval_boundaries(self):
        for start, stop, advances in (
            (1, 2, [1]),
            (0, 1, []),
            (2, 3, []),
            (1, 1, []),
        ):
            with self.subTest(start=start, stop=stop):
                results = search_ids(self.A, 489823, start=start, stop=stop)
                self.assertEqual([r.advance for r in results], advances)

    def test_duplicates_preserve_order_and_complete_traces(self):
        results = list(search_ids(self.F, 876352, stop=5))
        self.assertEqual([r.advance for r in results], [0, 1, 2, 3, 4])
        for advance, result in enumerate(results):
            with self.subTest(advance=advance):
                self.assertIsInstance(result, BDSPIDResult)
                self.assertEqual(result, BDSPIDResult(
                    advance, 0x80801000,
                    (0x80000000,) * (4 - advance) + (0x00801000,), self.F_END,
                ))
                self.assertEqual(result.calls_consumed, 5 - advance)

    def test_retries_extend_past_stop(self):
        self.assertEqual(
            list(search_ids(self.F, 876352, start=2, stop=3)),
            [BDSPIDResult(
                2, 0x80801000, (0x80000000, 0x80000000, 0x00801000), self.F_END
            )],
        )

    def test_leading_zero_match_retains_retry_trace(self):
        self.assertEqual(list(search_ids(self.D, 4096, stop=1)), [BDSPIDResult(
            0, 0x1000, (0x80000000, 0x80001000),
            (0, 0, 0x80000000, 0x80001000),
        )])

    def test_filter_is_lazy_and_returns_original_objects(self):
        originals = list(iter_ids(self.F, stop=5))
        draw = Mock(side_effect=originals)
        source = (draw() for _ in originals)
        with patch("core.bdsp.ids.iter_ids", return_value=source) as iterator:
            results = search_ids(self.F, 876352, stop=5)
            iterator.assert_called_once_with(self.F, start=0, stop=5)
            draw.assert_not_called()
            for original, result in zip(originals, results, strict=True):
                self.assertIs(result, original)
                self.assertIs(result.raw_outputs, original.raw_outputs)
                self.assertIs(result.state_after, original.state_after)

    def test_invalid_target_types_raise_immediately(self):
        for target in (True, False, 999999.0, "999999", "000000", None):
            for stop in (0, 1):
                with self.subTest(target=target, stop=stop):
                    with self.assertRaises(TypeError):
                        search_ids(self.A, target, stop=stop)

    def test_invalid_target_values_raise_immediately(self):
        for target in (-1, 1000000):
            for stop in (0, 1):
                with self.subTest(target=target, stop=stop):
                    with self.assertRaises(ValueError):
                        search_ids(self.A, target, stop=stop)

    def test_invalid_states_raise_immediately_including_empty_intervals(self):
        cases = (
            ((), ValueError),
            ((1, 2, 3), ValueError),
            ((1, 2, 3, 4, 5), ValueError),
            ((True, 2, 3, 4), TypeError),
            ((1, 2, False, 4), TypeError),
            ((1, 2, 3, 4.0), TypeError),
            ((1, "2", 3, 4), TypeError),
            ((-1, 2, 3, 4), ValueError),
            ((1, 2, 3, 0x100000000), ValueError),
            (None, TypeError),
        )
        for state, error in cases:
            for start, stop in ((0, 1), (1, 1)):
                with self.subTest(state=state, start=start, stop=stop):
                    with self.assertRaises(error):
                        search_ids(state, 489823, start=start, stop=stop)

    def test_invalid_intervals_raise_immediately(self):
        for value in (True, False, 1.0, "1", None):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    search_ids(self.A, 489823, start=value, stop=3)
                with self.assertRaises(TypeError):
                    search_ids(self.A, 489823, stop=value)
                with self.assertRaises(TypeError):
                    search_ids(self.A, 489823, start=value, stop=value)
        for start, stop in ((-1, 1), (0, -1), (-1, -1), (2, 1)):
            with self.subTest(start=start, stop=stop):
                with self.assertRaises(ValueError):
                    search_ids(self.A, 489823, start=start, stop=stop)

    def test_mutable_input_is_snapshotted_immediately(self):
        words = list(self.A)
        results = search_ids(words, 489823, stop=3)
        words[:] = [0, 0, 0, 0]
        self.assertEqual(list(results), [BDSPIDResult(
            1, 0x8000181F, (0x0000181F,), (3, 4, 0x0000080D, 0x0000181F)
        )])
        self.assertEqual(words, [0, 0, 0, 0])

    def test_input_is_not_mutated(self):
        words = list(self.F)
        results = search_ids(words, 876352, start=2, stop=3)
        self.assertEqual(words, list(self.F))
        self.assertEqual([r.advance for r in results], [2])
        self.assertEqual(words, list(self.F))


if __name__ == "__main__":
    unittest.main()
