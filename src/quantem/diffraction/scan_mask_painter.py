"""Select a 4D-STEM scan mask by dragging a region or a brush over the scan image.

The matplotlib scan-mask editor selects a region by *describing* it — a circle or
rectangle positioned with sliders. This one selects it by *pointing at it*, in
two modes:

``Region``
    The ROI box **is** the selection. Drag it to move the region and drag its
    handles to resize it — one side at a time, opposite side pinned, as in any
    drawing program — and the mask follows the outline live.
``Add`` / ``Erase``
    The ROI becomes a brush: drag it over the scan to add or remove probe
    positions, so the mask can follow whatever the scan actually shows instead
    of a shape.

It works by reusing ``quantem.widget``'s ``Show2D``. That widget is built on
anywidget (a real browser frontend, so it has genuine pointer events, unlike the
static-PNG matplotlib editor), and it already ships a draggable ROI. The relevant
detail is that its frontend updates ``roi_list`` from the pointer-MOVE handler
while a drag is in progress, not just on release — so the ROI streams into the
kernel continuously and each update can redraw the region or lay down ink.

The mask is shown by dimming the probe positions that are excluded, so the scan
image stays visible underneath.

Constructed through ``BraggPeaksPolymer.edit_scan_mask()``. State files are the
same format the matplotlib editor writes, so masks interchange between the two.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import ipywidgets as widgets
import numpy as np
from ipywidgets import Button, HBox, VBox
from IPython.display import display

from quantem.diffraction.scan_mask_editor import (
    GEOMETRIES,
    geometry_footprint,
    read_mask_state,
    resolve_reference_image,
    write_mask_state,
)

__all__ = ["ScanMaskPainter"]


class ScanMaskPainter:
    """Drag-to-select scan-mask editor backed by ``quantem.widget.Show2D``."""

    SCHEMA_VERSION = 3
    UNDO_LIMIT = 40
    # The frontend reports pointer moves but not pointer-up, so a stroke is
    # delimited by a pause instead. Everything inside one drag coalesces into a
    # single undo entry; without this, Undo would step back one pointer sample.
    STROKE_GAP_SECONDS = 0.35
    # ROI shape to open with per stored geometry. The widget's ROI has no
    # ellipse, so an elliptical seed opens as a circle; the seed *mask* keeps
    # its true shape until the ROI is first touched.
    ROI_SHAPE_FOR_GEOMETRY = {
        "circle": "circle",
        "ellipse": "circle",
        "square": "square",
        "rectangle": "rectangle",
    }

    def __init__(
        self,
        analysis,
        *,
        initial_x=None,
        initial_y=None,
        initial_radius=None,
        initial_geometry="circle",
        initial_size_x=None,
        initial_size_y=None,
        initial_angle=None,
        reference_image=None,
        state_path=None,
        autosave=False,
        display_widget=True,
        brush_radius=4,
        dim_factor=0.30,
        size=460,
    ):
        self.analysis = analysis
        self.scan_shape = tuple(int(v) for v in analysis.dataset_cartesian.shape[:2])
        self.state_path = None if state_path is None else Path(state_path)
        self.autosave = bool(autosave)
        self.dim_factor = float(dim_factor)
        self._dirty = False
        self._saved = False
        self._loaded = False
        self._undo_stack = []
        self._stroke_open = False
        self._last_paint_time = 0.0
        self._last_brush_pos = None
        self._last_roi_signature = None
        self._applying_roi = False

        rows, columns = self.scan_shape
        self.reference_image = self._resolve_reference_image(reference_image)
        self._reference_scaled = self._scale_reference(self.reference_image)

        geometry = str(initial_geometry).lower()
        if geometry not in GEOMETRIES:
            raise ValueError(
                f"Unknown mask geometry {geometry!r}; choose one of "
                f"{', '.join(GEOMETRIES)}."
            )
        center_row = rows // 2 if initial_y is None else int(initial_y)
        center_column = columns // 2 if initial_x is None else int(initial_x)
        radius = (
            max(1, min(rows, columns) // 3)
            if initial_radius is None
            else int(initial_radius)
        )
        size_x = radius if initial_size_x is None else int(initial_size_x)
        size_y = radius if initial_size_y is None else int(initial_size_y)
        angle = 0.0 if initial_angle is None else float(initial_angle) % 180.0

        if self.state_path is not None and self.state_path.is_file():
            state = read_mask_state(self.state_path, self.scan_shape, self.SCHEMA_VERSION)
            center_row = state["center_row"]
            center_column = state["center_column"]
            geometry = state["geometry"]
            size_x, size_y, angle = state["size_x"], state["size_y"], state["angle"]
            self._mask = state["mask"].copy()
            self._loaded = True
            self._saved = True
        else:
            self._mask = geometry_footprint(
                self.scan_shape, center_row, center_column, geometry, size_x, size_y, angle
            )

        self._seed_geometry = (
            center_column, center_row, geometry, size_x, size_y, angle
        )
        self._initial_mask = self._mask.copy()

        self.viewer = self._build_viewer(size=size)
        # Open the ROI on the stored region rather than on a bare brush dot, so
        # Region mode starts from the selection that was loaded instead of
        # replacing it with something unrelated on the first nudge.
        self.viewer.add_roi(
            row=center_row,
            col=center_column,
            shape=self.ROI_SHAPE_FOR_GEOMETRY.get(geometry, "circle"),
        )
        # The stored half-sizes count pixels *out from a centre pixel*, so they
        # describe 2h+1 positions, while an outline of width w covers w. The two
        # parameterisations cannot both be exact on an integer centre, so the
        # outline opens one position narrower than a mask the slider editor
        # saved; the first drag is what reconciles them. The mask array itself
        # round-trips untouched either way.
        if geometry == "rectangle":
            self.viewer._upsert_selected_roi(
                {"width": int(2 * size_x), "height": int(2 * size_y)}
            )
        else:
            self._set_brush_radius(int(size_x))
        self.brush_radius = int(brush_radius)
        # Seed the change guards from where the ROI actually starts. Left at
        # None, the first roi_list event would read as a change and repaint even
        # if it was only the radius slider.
        self._last_brush_pos = (int(center_row), int(center_column))
        self._last_roi_signature = self._roi_signature(self._selected_roi())
        self.viewer.observe(self._on_roi_change, names="roi_list")

        self.paint_selector = widgets.ToggleButtons(
            options=[
                ("Region", "region"),
                ("Add", "add"),
                ("Erase", "erase"),
                ("Off", "off"),
            ],
            value="region",
            description="Mode",
            tooltips=[
                "The ROI is the selection: drag it and its handles to shape the region",
                "Use the ROI as a brush that adds the probe positions it is dragged over",
                "Use the ROI as a brush that removes the probe positions it is dragged over",
                "Drag the ROI without changing the mask",
            ],
            style={"description_width": "44px", "button_width": "58px"},
        )
        self.apply_button = Button(
            description="Apply", button_style="primary",
            tooltip="Commit the selected mask to analysis.scan_mask",
            layout=widgets.Layout(width="76px"),
        )
        self.save_button = Button(
            description="Apply & Save", button_style="success",
            tooltip="Commit and write the state file for the next run",
            layout=widgets.Layout(width="112px"),
            disabled=self.state_path is None,
        )
        self.undo_button = Button(
            description="Undo", tooltip="Undo the last stroke or button action",
            layout=widgets.Layout(width="70px"),
        )
        self.full_button = Button(
            description="All", tooltip="Select every probe position",
            layout=widgets.Layout(width="56px"),
        )
        self.clear_button = Button(
            description="None", tooltip="Deselect every probe position",
            layout=widgets.Layout(width="60px"),
        )
        self.stamp_button = Button(
            description="Stamp",
            tooltip="Apply the ROI once where it stands, without moving it",
            layout=widgets.Layout(width="68px"),
        )
        self.reset_button = Button(
            description="Reset", tooltip="Restore the loaded/seed mask",
            layout=widgets.Layout(width="68px"),
        )
        self.status = widgets.HTML()

        self.apply_button.on_click(lambda _: self.apply())
        self.save_button.on_click(lambda _: self.save())
        self.undo_button.on_click(lambda _: self.undo())
        self.full_button.on_click(lambda _: self.set_all(True))
        self.clear_button.on_click(lambda _: self.set_all(False))
        self.stamp_button.on_click(lambda _: self.stamp())
        self.reset_button.on_click(lambda _: self.reset())
        self.paint_selector.observe(self._on_paint_action, names="value")

        self.widget = VBox(
            [
                HBox(
                    [
                        self.apply_button, self.save_button, self.undo_button,
                        self.full_button, self.clear_button, self.reset_button,
                    ],
                    layout=widgets.Layout(flex_flow="row wrap"),
                ),
                HBox([self.paint_selector, self.stamp_button],
                     layout=widgets.Layout(align_items="center")),
                self.viewer,
                self.status,
            ]
        )

        # A loaded/seed mask is usable immediately, as in the matplotlib editor.
        self.analysis.scan_mask = self._mask.copy()
        self._refresh_display()
        self._refresh_status(
            "Loaded saved mask" if self._loaded else "Seed mask applied"
        )
        if display_widget:
            display(self.widget)

    # -- construction helpers -------------------------------------------------

    def _build_viewer(self, *, size):
        from quantem.widget import Show2D

        # display_bin=1 keeps Show2D at full resolution, which matters because
        # _refresh_display writes the display array in place; under binning the
        # displayed array is a separate downsampled copy.
        return Show2D(
            self._display_array(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            title="Scan mask — drag the ROI to paint",
            labels=["scan"],
            show_fft=False,
            show_stats=False,
            show_controls=True,
            scale_bar_visible=False,
            display_bin=1,
            size=int(size),
            verbose=False,
        )

    def _resolve_reference_image(self, reference_image):
        return resolve_reference_image(self.analysis, reference_image, self.scan_shape)

    @staticmethod
    def _scale_reference(image):
        """Normalise to 0..1 on 1/99 percentiles so the dimming reads consistently."""
        finite = image[np.isfinite(image)]
        if finite.size:
            low, high = np.percentile(finite, [1.0, 99.0])
            if not high > low:
                low, high = float(finite.min()), float(finite.max() + 1.0)
        else:
            low, high = 0.0, 1.0
        scaled = (np.nan_to_num(image, nan=low) - low) / (high - low)
        return np.clip(scaled, 0.0, 1.0).astype(np.float32)

    # -- display --------------------------------------------------------------

    def _display_array(self):
        """Reference image with excluded probe positions dimmed."""
        weighted = self._reference_scaled * np.where(
            self._mask, 1.0, self.dim_factor
        ).astype(np.float32)
        return weighted[np.newaxis, ...].astype(np.float32)

    def _refresh_display(self):
        # Show2D exposes no public setter for pixel data, so write the display
        # array in place and ask it to re-send the frame. In place matters: when
        # unbinned, Show2D's _display_data is the same object as _data.
        try:
            self.viewer._data[0][:] = self._display_array()[0]
            self.viewer._update_all_frames()
        except Exception as exc:  # pragma: no cover - frontend/version drift
            self.status.value = (
                f"<b>Mask updated, but the preview could not refresh</b> ({exc}). "
                "The mask itself is correct; Apply still works."
            )

    def _refresh_status(self, message):
        count = int(self._mask.sum())
        total = int(self._mask.size)
        physical = ""
        try:
            row_sampling, column_sampling = (
                float(v) for v in self.analysis.dataset_cartesian.sampling[:2]
            )
            row_unit, column_unit = (
                str(v) for v in self.analysis.dataset_cartesian.units[:2]
            )
            if np.isclose(row_sampling, column_sampling) and row_unit == column_unit:
                area = count * row_sampling * column_sampling
                physical = f" · {area:.4g} {row_unit}²"
        except (TypeError, ValueError):
            pass
        brush = self._roi_description()
        self.status.value = (
            f"<b>{message}</b><br>"
            f"{brush} · {count:,}/{total:,} positions "
            f"({100.0 * count / total:.1f}%){physical} · "
            f"{'saved' if self._saved else 'not saved'}"
            f"{' · unapplied edits' if self._dirty else ''}"
        )

    # -- brush ----------------------------------------------------------------

    def _selected_roi(self):
        rois = list(self.viewer.roi_list or [])
        index = int(self.viewer.roi_selected_idx)
        if not 0 <= index < len(rois):
            return None
        return rois[index]

    def _set_brush_radius(self, radius):
        self.viewer._upsert_selected_roi({"radius": int(max(1, radius))})

    @staticmethod
    def _round_half_up(value):
        """Round halves upward, the way the frontend's ``Math.round`` does.

        Resizing one side of an odd-sized ROI leaves its centre on a half
        pixel, so halves are the common case here rather than the rare one.
        ``round()`` would break them to even and disagree with the outline the
        widget drew; ``int()`` would bias every one towards the origin.
        """
        return int(math.floor(float(value) + 0.5))

    @classmethod
    def _roi_center(cls, roi):
        """ROI centre as whole scan positions."""
        return (
            cls._round_half_up(roi.get("row", 0)),
            cls._round_half_up(roi.get("col", 0)),
        )

    @staticmethod
    def _roi_signature(roi):
        """Position *and* size of the ROI, so a resize counts as a change.

        Region mode has to react to a resize, which leaves the centre alone —
        watching the centre alone is what makes a handle drag look like nothing
        is happening.
        """
        if roi is None:
            return None
        return tuple(
            roi.get(key)
            for key in ("row", "col", "shape", "radius", "radius_inner", "width", "height")
        )

    def _roi_description(self):
        roi = self._selected_roi()
        noun = "region" if self.paint_selector.value == "region" else "brush"
        if roi is None:
            return f"no {noun}"
        shape = str(roi.get("shape", "circle"))
        if shape == "rectangle":
            return f"{int(roi.get('width', 0))}×{int(roi.get('height', 0))} px {noun}"
        if shape == "annular":
            return (
                f"annular {noun} r={int(roi.get('radius_inner', 0))}–"
                f"{int(roi.get('radius', 0))} px"
            )
        if shape == "square":
            size = 2 * int(roi.get("radius", 0))
            return f"{size}×{size} px square {noun}"
        return f"circular {noun} r={int(roi.get('radius', 0))} px"

    def _roi_footprint(self, roi, row=None, column=None):
        """Probe positions the ROI covers, matching the outline the widget draws.

        The widget's image coordinates put integers on pixel *boundaries*, so a
        position belongs to the ROI when its pixel centre does. Offsetting the
        centre by half a pixel is what lets ``geometry_footprint`` — which
        measures from pixel indices — reproduce the drawn outline instead of a
        footprint one row and one column too wide.
        """
        shape = str(roi.get("shape", "circle"))
        row = (float(roi.get("row", 0)) if row is None else float(row)) - 0.5
        column = (float(roi.get("col", 0)) if column is None else float(column)) - 0.5
        if shape == "rectangle":
            return geometry_footprint(
                self.scan_shape, row, column, "rectangle",
                max(0.5, float(roi.get("width", 2)) / 2.0),
                max(0.5, float(roi.get("height", 2)) / 2.0),
            )
        outer = max(0.5, float(roi.get("radius", 1)))
        if shape == "square":
            return geometry_footprint(
                self.scan_shape, row, column, "square", outer, outer
            )
        footprint = geometry_footprint(
            self.scan_shape, row, column, "circle", outer, outer
        )
        if shape == "annular":
            inner = max(0.0, float(roi.get("radius_inner", 0)))
            if inner:
                footprint = footprint & ~geometry_footprint(
                    self.scan_shape, row, column, "circle", inner, inner
                )
        return footprint

    def _swept_footprint(self, roi, start, end):
        """Footprint of the ROI dragged from ``start`` to ``end``, gaps filled in.

        Pointer samples reach the kernel coarsely: the browser coalesces moves
        and the round trip drops more, so a quick drag arrives as a handful of
        widely separated positions. Stamping only those positions leaves a
        dotted trail — the stroke the user drew comes out as disconnected
        blobs — so the footprint is swept along the segment between samples.
        The intermediate stamps differ from the one under the pointer only by a
        whole-position offset, so they are shifted rather than recomputed.
        """
        footprint = self._roi_footprint(roi)
        steps = int(max(abs(end[0] - start[0]), abs(end[1] - start[1])))
        if steps <= 1:
            return footprint
        rows, columns = self.scan_shape
        swept = footprint.copy()
        for step in range(steps):
            fraction = step / steps
            row_shift = round(start[0] + (end[0] - start[0]) * fraction) - end[0]
            column_shift = round(start[1] + (end[1] - start[1]) * fraction) - end[1]
            source = (
                slice(max(0, -row_shift), rows - max(0, row_shift)),
                slice(max(0, -column_shift), columns - max(0, column_shift)),
            )
            target = (
                slice(max(0, row_shift), rows - max(0, -row_shift)),
                slice(max(0, column_shift), columns - max(0, -column_shift)),
            )
            swept[target] |= footprint[source]
        return swept

    # -- painting -------------------------------------------------------------

    def _push_undo(self):
        self._undo_stack.append(self._mask.copy())
        del self._undo_stack[: -self.UNDO_LIMIT]

    def _on_roi_change(self, _change):
        roi = self._selected_roi()
        if roi is None or self._applying_roi:
            return
        position = self._roi_center(roi)
        signature = self._roi_signature(roi)
        previous_position = self._last_brush_pos
        reshaped = signature != self._last_roi_signature
        moved = position != previous_position
        self._last_brush_pos = position
        self._last_roi_signature = signature
        action = self.paint_selector.value
        if action == "off" or not reshaped:
            return
        self._begin_stroke()
        if action == "region":
            # The outline *is* the selection, so a resize has to land as well as
            # a move: this is the drag that the centre-only guard below used to
            # swallow.
            self._set_region(self._roi_footprint(roi))
            return
        if not moved:
            # roi_list also fires when the brush is resized or reshaped from the
            # widget's own controls. Painting then would drop a blob every time
            # the radius slider moved, so only a change of position lays ink.
            # Use Stamp to paint without moving.
            self._refresh_status(f"Brush {action} — {self._roi_description()}")
            return
        start = previous_position if previous_position is not None else position
        self.paint(
            self._swept_footprint(roi, start, position), erase=action == "erase"
        )

    def _begin_stroke(self):
        """Close the previous drag if this event follows a pause.

        The frontend reports pointer moves but never pointer-up, so a drag is
        delimited by a pause instead. Everything inside one drag then shares a
        single undo entry, pushed by whichever of ``paint``/``_set_region``
        first has a real change to make.
        """
        now = time.monotonic()
        if now - self._last_paint_time > self.STROKE_GAP_SECONDS:
            self._stroke_open = False
        self._last_paint_time = now

    def _set_region(self, footprint, *, new_stroke=False):
        """Replace the mask with ``footprint``, the Region-mode commit."""
        footprint = np.asarray(footprint, dtype=bool)
        if np.array_equal(self._mask, footprint):
            return self
        if new_stroke or not self._stroke_open:
            self._push_undo()
            self._stroke_open = True
        self._mask = footprint.copy()
        self._dirty = True
        self._saved = False
        self._refresh_display()
        self._refresh_status(
            f"Region selected ({self._roi_description()}) — click Apply when done"
        )
        return self

    def stamp(self):
        """Apply the ROI once where it stands, without moving it."""
        action = self.paint_selector.value
        roi = self._selected_roi()
        if action == "off" or roi is None:
            self._refresh_status("Choose Region, Add or Erase before stamping")
            return self
        if action == "region":
            return self._set_region(self._roi_footprint(roi), new_stroke=True)
        return self.paint(
            self._roi_footprint(roi), erase=action == "erase", new_stroke=True
        )

    def paint(self, footprint, *, erase=False, new_stroke=False):
        """Composite a boolean footprint into the mask, one undo entry per stroke."""
        footprint = np.asarray(footprint, dtype=bool)
        if footprint.shape != self.scan_shape:
            raise ValueError(
                f"footprint shape {footprint.shape} must match scan shape "
                f"{self.scan_shape}."
            )
        if erase:
            changed = bool((self._mask & footprint).any())
        else:
            changed = bool((~self._mask & footprint).any())
        if not changed:
            return self
        if new_stroke or not self._stroke_open:
            self._push_undo()
            self._stroke_open = True
        if erase:
            self._mask &= ~footprint
        else:
            self._mask |= footprint
        self._dirty = True
        self._saved = False
        self._refresh_display()
        self._refresh_status(
            f"{'Erasing' if erase else 'Adding'} — click Apply when done"
        )
        return self

    def _on_paint_action(self, change):
        self._stroke_open = False
        entering, leaving = change["new"], change["old"]
        if leaving in {"add", "erase"}:
            # Remember how big the brush was left, so switching away and back
            # does not undo a deliberate resize.
            roi = self._selected_roi()
            if roi is not None and str(roi.get("shape")) != "rectangle":
                self.brush_radius = max(1, int(roi.get("radius", self.brush_radius)))
        if entering in {"add", "erase"} and leaving == "region":
            # A region-sized outline is useless as a brush, so entering a paint
            # mode brings the ROI back down to brush size.
            self._resize_roi_quietly({
                "shape": "circle", "radius": int(self.brush_radius)
            })
        if entering == "region":
            self._refresh_status(
                "Region mode — drag the ROI and its handles to shape the selection"
            )
        elif entering == "off":
            self._refresh_status("Off — drag the ROI without changing the mask")
        else:
            self._refresh_status(
                f"Brush {entering} — drag the ROI over the scan to paint"
            )

    def _resize_roi_quietly(self, updates):
        """Change the ROI without the change reading as a drag of the user's."""
        self._applying_roi = True
        try:
            self.viewer._upsert_selected_roi(updates)
        finally:
            self._applying_roi = False
        roi = self._selected_roi()
        self._last_roi_signature = self._roi_signature(roi)
        if roi is not None:
            self._last_brush_pos = self._roi_center(roi)

    # -- mask operations ------------------------------------------------------

    def set_all(self, selected):
        self._push_undo()
        self._stroke_open = False
        self._mask = np.full(self.scan_shape, bool(selected), dtype=bool)
        self._dirty = True
        self._saved = False
        self._refresh_display()
        self._refresh_status(
            "All positions selected" if selected else "All positions deselected"
        )
        return self

    def reset(self):
        self._push_undo()
        self._stroke_open = False
        self._mask = self._initial_mask.copy()
        self._dirty = True
        self._saved = False
        self._refresh_display()
        self._refresh_status("Seed mask restored; click Apply")
        return self

    def undo(self):
        if not self._undo_stack:
            self._refresh_status("Nothing to undo")
            return self
        self._mask = self._undo_stack.pop()
        self._stroke_open = False
        self._dirty = True
        self._saved = False
        self._refresh_display()
        self._refresh_status(f"Undone; {len(self._undo_stack)} step(s) left")
        return self

    def set_mask(self, mask):
        """Replace the mask wholesale, e.g. from a computed selection."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.scan_shape:
            raise ValueError(
                f"mask shape {mask.shape} must match scan shape {self.scan_shape}."
            )
        self._push_undo()
        self._stroke_open = False
        self._mask = mask.copy()
        self._dirty = True
        self._saved = False
        self._refresh_display()
        self._refresh_status("Mask set programmatically; click Apply")
        return self

    # -- commit ---------------------------------------------------------------

    def _stored_geometry(self):
        """Geometry fields for the state file: ``(column, row, geometry, x, y, angle)``.

        In Region mode the outline is the mask, so the ROI describes it exactly
        and is worth recording — reopening the file then puts the outline back
        where it was. A painted mask has no describing shape, so the seed is
        kept (the mask array is what actually round-trips either way).
        """
        roi = self._selected_roi()
        if self.paint_selector.value != "region" or roi is None:
            return self._seed_geometry
        shape = str(roi.get("shape", "circle"))
        if shape == "rectangle":
            size_x = max(1, self._round_half_up(float(roi.get("width", 2)) / 2))
            size_y = max(1, self._round_half_up(float(roi.get("height", 2)) / 2))
        elif shape in {"square", "circle"}:
            size_x = size_y = max(1, self._round_half_up(roi.get("radius", 1)))
        else:
            # An annulus is not one of the mask geometries, so fall back to the
            # seed rather than write a shape that cannot be read back.
            return self._seed_geometry
        row, column = self._roi_center(roi)
        return (column, row, shape, size_x, size_y, 0.0)

    def apply(self):
        self.analysis.scan_mask = self._mask.copy()
        self._dirty = False
        self._refresh_status("Mask applied")
        if self.autosave and self.state_path is not None:
            self.save(apply_first=False)
        return self.applied_mask

    def save(self, path=None, *, apply_first=True):
        path = self.state_path if path is None else Path(path)
        if path is None:
            raise ValueError("No scan-mask state_path was configured.")
        if apply_first:
            self.apply()
        column, row, geometry, size_x, size_y, angle = self._stored_geometry()
        write_mask_state(
            path,
            mask=self._mask,
            geometry=geometry,
            center_row=row,
            center_column=column,
            radius=size_x,
            size_x=size_x,
            size_y=size_y,
            angle=angle,
            scan_shape=self.scan_shape,
            sampling=self.analysis.dataset_cartesian.sampling[:2],
            units=self.analysis.dataset_cartesian.units[:2],
            schema_version=self.SCHEMA_VERSION,
        )
        self.state_path = path
        self.save_button.disabled = False
        self._saved = True
        self._dirty = False
        self._refresh_status(f"Applied and saved to {path}")
        return path

    def close(self):
        """Release the widget, matching ``ScanMaskEditor.close()``.

        The painter holds no Matplotlib figure, so there is nothing to shut
        down but the frontend; callers (and tests) close both editors the same
        way and should not have to know which backend they were handed.

        Closing twice is a no-op, since ``close()`` is the kind of thing a
        ``finally`` block calls after an error may already have closed it.
        """
        try:
            self.viewer.unobserve(self._on_roi_change, names="roi_list")
        except ValueError:
            return
        self.widget.close()
        self.viewer.close()

    # -- array-like access, matching ScanMaskEditor ---------------------------

    @property
    def mask(self):
        return self._mask

    @property
    def applied_mask(self):
        return None if self.analysis.scan_mask is None else self.analysis.scan_mask

    @property
    def shape(self):
        return self._mask.shape

    @property
    def dtype(self):
        return self._mask.dtype

    @property
    def size(self):
        return self._mask.size

    @property
    def ndim(self):
        return self._mask.ndim

    def __array__(self, dtype=None, copy=None):
        array = self._mask if dtype is None else self._mask.astype(dtype)
        return array.copy() if copy else array

    def __len__(self):
        return len(self._mask)

    def sum(self, *args, **kwargs):
        return self._mask.sum(*args, **kwargs)

    def astype(self, *args, **kwargs):
        return self._mask.astype(*args, **kwargs)

    def copy(self):
        return self._mask.copy()

    def __getitem__(self, key):
        return self._mask[key]
