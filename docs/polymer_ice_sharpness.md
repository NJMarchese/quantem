# Ice sharpness: core and envelope widths

The ice detector measures two FWHM values from the same baseline-adjusted,
Gaussian-smoothed radial and annular cuts:

- **Core FWHM** stops at the first half-height crossing on each side. Smoothing
  remains enabled, so a one-bin noise fluctuation is less likely to create a
  false narrow core.
- **Envelope FWHM** requires the crossing to remain below half height for
  `sharpness_crossing_persistence` bins. This is the historical `width_*`
  measurement and follows wave-like profiles across shallow troughs and side
  lobes.
- **Envelope/core ratio** is near 1 for a smooth single peak and greater than 1
  when a narrow core sits inside a broader or oscillatory envelope.

## Where the criteria run

Candidate filtering is ordered as follows:

1. Keep peaks in the configured ice q band.
2. Apply every configured envelope ceiling, core ceiling, and ratio floor.
3. Apply the intensity threshold.
4. Fit the six-fold ice geometry to the survivors.

Thus a core-width criterion is an early appearance gate, not a replacement for
the crystallographic geometry check. Broad polymer peaks removed at step 2 cannot
influence the fitted ice orientation.

## Recommended use for wave-like annular peaks

To admit both conventional sharp peaks and sharp cores with broad side lobes,
turn off the annular envelope ceiling and gate on the annular core:

```python
import dataclasses

params_ice_flagger = dataclasses.replace(
    params_ice_flagger,
    max_width_theta_deg=None,
    max_core_width_theta_deg=8.0,
    min_envelope_core_ratio_theta=None,
)
```

Do not leave `max_width_theta_deg` enabled if the purpose is to rescue a sharp
core inside a broad envelope: all configured sharpness criteria are ANDed.

Use a ratio floor only when intentionally selecting the wave-like subset:

```python
params_wave_like = dataclasses.replace(
    params_ice_flagger,
    min_envelope_core_ratio_theta=1.5,
)
```

A ratio floor rejects ordinary sharp peaks whose core and envelope widths are
similar, so it is normally a diagnostic rather than the main all-ice gate.
`selection_box(...)` plots annular core against envelope and returns the tuned
parameters. `measure_ice_peak_widths()` returns `core_width_*` and
`envelope_core_ratio_*` arrays for choosing thresholds from the actual scan.

