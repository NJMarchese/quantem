"""Region/brush behaviour of the Show2D-backed scan-mask painter.

The frontend is a browser, so the drags are simulated the way it drives the
kernel: by writing a new ``roi_list`` onto the viewer, which is exactly what its
pointer handlers do.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from quantem.diffraction import BraggPeaksPolymer

ScanMaskPainter = pytest.importorskip(
    "quantem.diffraction.scan_mask_painter"
).ScanMaskPainter


def _analysis(scan_shape=(40, 60)):
    rows, columns = scan_shape
    analysis = object.__new__(BraggPeaksPolymer)
    rng = np.random.default_rng(0)
    analysis.dataset_cartesian = SimpleNamespace(
        shape=(rows, columns, 4, 4),
        array=rng.random((rows, columns, 4, 4)),
        sampling=(0.02, 0.02, 1.0, 1.0),
        units=("µm", "µm", "1/Å", "1/Å"),
        virtual_images={},
    )
    analysis._scan_mask = None
    return analysis


def _painter(**kwargs):
    return ScanMaskPainter(_analysis(), display_widget=False, **kwargs)


def _drag_roi(painter, **updates):
    """Update the ROI the way the frontend does mid-drag."""
    rois = [dict(painter.viewer.roi_list[0])]
    rois[0].update(updates)
    painter.viewer.roi_list = rois


def test_region_mode_follows_a_resize_that_does_not_move_the_centre():
    painter = _painter(initial_geometry="rectangle", initial_size_x=8, initial_size_y=5)
    try:
        assert painter.paint_selector.value == "region"
        _drag_roi(painter, row=20, col=30, width=20, height=10)
        before = painter.mask.copy()
        # Dragging a side leaves the centre alone, which is the change a
        # position-only guard swallows.
        _drag_roi(painter, width=30, col=35)
        assert not np.array_equal(painter.mask, before)
        columns_hit = np.flatnonzero(painter.mask.any(axis=0))
        # The left side was not grabbed, so it has not moved.
        assert columns_hit.min() == 20
        assert columns_hit.max() == 49
    finally:
        painter.close()


def test_region_footprint_matches_the_outline_the_widget_draws():
    painter = _painter(initial_geometry="rectangle")
    try:
        _drag_roi(painter, shape="rectangle", row=20, col=30, width=20, height=10)
        rows_hit = np.flatnonzero(painter.mask.any(axis=1))
        columns_hit = np.flatnonzero(painter.mask.any(axis=0))
        # A 20x10 box selects 20x10 positions, not 21x11.
        assert (columns_hit.size, rows_hit.size) == (20, 10)
        assert painter.mask.sum() == 200
    finally:
        painter.close()


def test_a_square_roi_selects_a_square():
    painter = _painter()
    try:
        _drag_roi(painter, shape="square", radius=5, row=20, col=30)
        assert painter.mask.sum() == 100
        rows_hit = np.flatnonzero(painter.mask.any(axis=1))
        assert rows_hit.size == 10
        # Every row of the square spans the same columns; a circle would not.
        spans = {int(row.sum()) for row in painter.mask[rows_hit]}
        assert spans == {10}
    finally:
        painter.close()


def test_a_circular_roi_selects_a_disc():
    painter = _painter()
    try:
        _drag_roi(painter, shape="circle", radius=6, row=20, col=30)
        assert painter.mask.sum() == pytest.approx(np.pi * 36, rel=0.05)
        rows_hit = np.flatnonzero(painter.mask.any(axis=1))
        widths = [int(painter.mask[row].sum()) for row in rows_hit]
        assert widths[0] < max(widths)
    finally:
        painter.close()


def test_one_drag_is_one_undo_entry_and_undo_restores_the_region():
    painter = _painter()
    try:
        original = painter.mask.copy()
        depth = len(painter._undo_stack)
        for column in (10, 12, 14, 16):
            _drag_roi(painter, row=20, col=column)
        assert len(painter._undo_stack) - depth == 1
        painter.undo()
        np.testing.assert_array_equal(painter.mask, original)
    finally:
        painter.close()


def test_a_brush_drag_lays_down_a_continuous_stroke():
    painter = _painter(initial_radius=1)
    try:
        painter.paint_selector.value = "add"
        painter.set_all(False)
        # Three widely separated pointer samples, as a quick drag arrives once
        # the browser has coalesced its moves.
        for column in (5, 25, 50):
            _drag_roi(painter, row=20, col=column)
        painted = np.flatnonzero(painter.mask[20])
        assert painted.size
        # No gaps: the stroke is one connected run, not three blobs.
        assert np.all(np.diff(painted) == 1)
        assert painted.min() <= 5 and painted.max() >= 50
    finally:
        painter.close()


def test_erasing_sweeps_the_same_path_it_would_have_painted():
    painter = _painter()
    try:
        painter.set_all(True)
        painter.paint_selector.value = "erase"
        for column in (5, 25, 50):
            _drag_roi(painter, row=20, col=column)
        cleared = np.flatnonzero(~painter.mask[20])
        assert np.all(np.diff(cleared) == 1)
    finally:
        painter.close()


def test_entering_a_paint_mode_brings_the_roi_down_to_brush_size():
    painter = _painter(initial_geometry="rectangle", initial_size_x=20, initial_size_y=15,
                       brush_radius=3)
    try:
        painter.paint_selector.value = "add"
        roi = painter.viewer.roi_list[0]
        assert (roi["shape"], roi["radius"]) == ("circle", 3)
        # A resize made while painting is remembered across a mode round trip.
        _drag_roi(painter, radius=7)
        painter.paint_selector.value = "region"
        painter.paint_selector.value = "erase"
        assert painter.viewer.roi_list[0]["radius"] == 7
    finally:
        painter.close()


def test_off_mode_leaves_the_mask_alone():
    painter = _painter()
    try:
        painter.paint_selector.value = "off"
        frozen = painter.mask.copy()
        _drag_roi(painter, row=30, col=30, radius=9, shape="circle")
        np.testing.assert_array_equal(painter.mask, frozen)
        assert not painter._dirty
    finally:
        painter.close()


def test_preview_is_only_committed_on_apply():
    analysis = _analysis()
    painter = ScanMaskPainter(analysis, display_widget=False)
    try:
        seeded = analysis.scan_mask.copy()
        _drag_roi(painter, shape="rectangle", row=20, col=30, width=20, height=10)
        assert not np.array_equal(painter.mask, seeded)
        np.testing.assert_array_equal(analysis.scan_mask, seeded)
        painter.apply()
        np.testing.assert_array_equal(analysis.scan_mask, painter.mask)
    finally:
        painter.close()


def test_region_mode_saves_the_outline_so_reopening_restores_it(tmp_path):
    path = tmp_path / "scan_mask.npz"
    analysis = _analysis()
    painter = ScanMaskPainter(analysis, display_widget=False, state_path=path)
    try:
        _drag_roi(painter, shape="rectangle", row=20, col=30, width=20, height=10)
        expected = painter.mask.copy()
        painter.save()
        np.testing.assert_array_equal(analysis.scan_mask, expected)
    finally:
        painter.close()

    with np.load(path, allow_pickle=False) as state:
        assert str(state["geometry"].item()) == "rectangle"
        assert (int(state["size_x"]), int(state["size_y"])) == (10, 5)
        assert (int(state["center_row"]), int(state["center_column"])) == (20, 30)

    reopened = ScanMaskPainter(_analysis(), display_widget=False, state_path=path)
    try:
        roi = reopened.viewer.roi_list[0]
        assert (roi["shape"], roi["row"], roi["col"], roi["width"], roi["height"]) == (
            "rectangle", 20, 30, 20, 10,
        )
        np.testing.assert_array_equal(reopened.mask, expected)
        # The reopened outline reselects the same positions it was saved from.
        _drag_roi(reopened, col=30)
        np.testing.assert_array_equal(reopened.mask, expected)
    finally:
        reopened.close()


def test_a_painted_mask_keeps_its_seed_geometry_in_the_state_file(tmp_path):
    path = tmp_path / "scan_mask.npz"
    painter = ScanMaskPainter(
        _analysis(), display_widget=False, state_path=path,
        initial_geometry="ellipse", initial_size_x=9, initial_size_y=4,
    )
    try:
        painter.paint_selector.value = "add"
        _drag_roi(painter, row=5, col=5)
        painter.save()
    finally:
        painter.close()
    with np.load(path, allow_pickle=False) as state:
        # No ROI describes a painted mask, so the seed shape is what is recorded
        # and the mask array is what actually round-trips.
        assert str(state["geometry"].item()) == "ellipse"
        assert (int(state["size_x"]), int(state["size_y"])) == (9, 4)


def test_a_half_pixel_roi_centre_still_selects_the_drawn_box():
    """Resizing one side of an odd-sized ROI puts its centre on a half pixel.

    That is what lets the anchored side stay exactly put, so the mask has to
    read those centres as drawn rather than truncating them.
    """
    painter = _painter(initial_geometry="rectangle")
    try:
        # A 41x11 box anchored with its left side on column 10 and its top on
        # row 15, exactly as the frontend's one-sided resize reports it.
        _drag_roi(painter, shape="rectangle", row=20.5, col=30.5, width=41, height=11)
        columns_hit = np.flatnonzero(painter.mask.any(axis=0))
        rows_hit = np.flatnonzero(painter.mask.any(axis=1))
        assert (columns_hit.min(), columns_hit.max()) == (10, 50)
        assert (rows_hit.min(), rows_hit.max()) == (15, 25)
        assert painter.mask.sum() == 41 * 11
    finally:
        painter.close()


def test_a_half_pixel_brush_centre_paints_where_it_is_drawn():
    painter = _painter(initial_radius=1)
    try:
        painter.paint_selector.value = "add"
        painter.set_all(False)
        _drag_roi(painter, shape="square", row=20.5, col=30.5, radius=2.5)
        rows_hit = np.flatnonzero(painter.mask.any(axis=1))
        columns_hit = np.flatnonzero(painter.mask.any(axis=0))
        assert (rows_hit.min(), rows_hit.max()) == (18, 22)
        assert (columns_hit.min(), columns_hit.max()) == (28, 32)
    finally:
        painter.close()


def test_a_half_pixel_region_round_trips_through_the_state_file(tmp_path):
    path = tmp_path / "scan_mask.npz"
    painter = ScanMaskPainter(_analysis(), display_widget=False, state_path=path)
    try:
        _drag_roi(painter, shape="rectangle", row=20.5, col=30.5, width=41, height=11)
        expected = painter.mask.copy()
        painter.save()
    finally:
        painter.close()
    with np.load(path, allow_pickle=False) as state:
        # Half-sizes round rather than truncate, so the recorded shape is the
        # nearest describable one instead of one pixel small.
        assert (int(state["size_x"]), int(state["size_y"])) == (21, 6)
        assert (int(state["center_row"]), int(state["center_column"])) == (21, 31)
        np.testing.assert_array_equal(np.asarray(state["mask"], dtype=bool), expected)


def test_close_is_safe_to_call_twice():
    painter = _painter()
    painter.close()
    painter.close()
