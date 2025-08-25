# To run this script you need Python 3.9+ installed.
# Required packages: numpy, plotly, astropy (and optionally ipywidgets for the date picker).
# Install them via:
#   pip install numpy plotly astropy ipywidgets
#
# Save this script as e.g. solar_system.py and run with:
#   python solar_system.py
# It will generate solar_system_plan_view.html which you can open in your browser.

import math
import warnings
from datetime import datetime, date as date_cls
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from astropy.time import Time
from astropy.coordinates import get_body_barycentric
from astropy import units as u
from erfa import ErfaWarning

# Optional notebook UI; safe to run without
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML
    HAVE_WIDGETS = True
except Exception:
    HAVE_WIDGETS = False

"""
Solar System plan-view (inner + outer panels) with full orbits and ±15-day motion arcs.
- Inner panel: Mercury→Mars.  Outer panel: Jupiter→Neptune.
- Date entry box (if ipywidgets available); otherwise edit date_str below.
- Full orbits: thin faint lines.  ±15-day arcs: thicker segments centred on the date.
- Coordinates are HELIOCENTRIC **in the ecliptic plane** (Sun fixed at 0,0).

RENDERING NOTE:
Some environments do not have nbformat installed, which Plotly's default renderer
requires. To avoid the error `ValueError: Mime type rendering requires nbformat>=4.2.0`,
this script **does not call fig.show()**. Instead it embeds HTML directly when possible
or saves a standalone HTML file you can open in any browser.

DATE RANGE & WARNINGS:
- Precise ephemerides are used within ~1900–2100.
- Outside that, we fall back to an **elliptical, planar Kepler model** per planet
  using (a, e, ϖ, L₀ at J2000) to preserve realistic perihelion/aphelion and orientation.
- ERFA "dubious year" warnings are silenced for readability (they're harmless).
"""

# --------------------------
# Warning control (silence ERFA 'dubious year' chatter)
# --------------------------
warnings.filterwarnings("ignore", category=ErfaWarning)

# --------------------------
# Configuration
# --------------------------
INNER_PLANETS = ["mercury", "venus", "earth", "mars"]
OUTER_PLANETS = ["jupiter", "saturn", "uranus", "neptune"]
HELIOCENTRIC = True          # keep Sun at (0,0)
ARC_DAYS = 15                # ± days for the arc
ARC_SAMPLES = 100            # points along the arc
ARC_LINE_WIDTH = 3           # thicker line for the arc
ORBIT_SAMPLES = 500          # resolution of the full orbit line
ORBIT_LINE_WIDTH = 1         # thin line for full orbit
ORBIT_LINE_COLOR = "rgba(150,150,150,0.5)"  # faint grey
OUTPUT_HTML = "solar_system_plan_view.html"

# Distinct colours per planet and marker sizes
PLANET_COLOR = {
    "mercury": "#8c8c8c",  # grey
    "venus":   "#d98e04",  # ochre
    "earth":   "#1f77b4",  # blue
    "mars":    "#d62728",  # red
    "jupiter": "#9467bd",  # purple-brown
    "saturn":  "#bcbd22",  # olive
    "uranus":  "#17becf",  # cyan
    "neptune": "#2ca02c",  # green-blue
}
MARKER_SIZE = {
    "mercury": 8,
    "venus": 9,
    "earth": 10,
    "mars": 9,
    "jupiter": 12,
    "saturn": 11,
    "uranus": 10,
    "neptune": 10,
}

# Sidereal orbital periods (days)
ORBITAL_PERIOD_DAYS = {
    "mercury": 87.9691,
    "venus": 224.701,
    "earth": 365.256,
    "mars": 686.980,
    "jupiter": 4332.589,
    "saturn": 10759.22,
    "uranus": 30685.4,
    "neptune": 60189.0,
}

# Date range for precise ephemerides (loose guard to avoid spurious ERFA warnings)
PRECISE_MIN = datetime(1900, 1, 1)
PRECISE_MAX = datetime(2100, 12, 31)

# Elliptical fallback parameters (J2000). Values are typical VSOP-style constants.
SEMI_MAJOR_AU = {
    "mercury": 0.387098,
    "venus": 0.723332,
    "earth": 1.000000,
    "mars": 1.523679,
    "jupiter": 5.20260,
    "saturn": 9.55491,
    "uranus": 19.2184,
    "neptune": 30.1104,
}
ECCENTRICITY = {
    "mercury": 0.205630,
    "venus": 0.006772,
    "earth": 0.016711,
    "mars": 0.09341,
    "jupiter": 0.0489,
    "saturn": 0.0565,
    "uranus": 0.0457,
    "neptune": 0.0113,
}
VARPI_DEG = {
    "mercury": 77.456119,
    "venus": 131.563707,
    "earth": 102.937682,
    "mars": 336.04084,
    "jupiter": 14.75385,
    "saturn": 92.43194,
    "uranus": 170.96424,
    "neptune": 44.97135,
}
L0_DEG = {
    "mercury": 252.250906,
    "venus": 181.979801,
    "earth": 100.464572,
    "mars": 355.45332,
    "jupiter": 34.39644,
    "saturn": 49.954244,
    "uranus": 313.238104,
    "neptune": 304.879970,
}

J2000_TDB = Time("2000-01-01T12:00:00", scale="tdb")

# Obliquity of the ecliptic (J2000) for equatorial→ecliptic rotation (radians)
_OBLIQ_DEG = 23.439291111
_C = math.cos(math.radians(_OBLIQ_DEG))
_S = math.sin(math.radians(_OBLIQ_DEG))

# --------------------------
# Core calculations
# --------------------------
def _use_precise(date):
    """Return True if date within PRECISE_MIN..PRECISE_MAX (inclusive)."""
    if isinstance(date, str):
        try:
            dt = datetime.strptime(date.strip(), "%Y-%m-%d")
        except Exception:
            try:
                dt = datetime.fromisoformat(date.strip().split("T")[0])
            except Exception:
                return True
    elif isinstance(date, (datetime, date_cls)):
        dt = date if isinstance(date, datetime) else datetime(date.year, date.month, date.day)
    else:
        return True
    return (PRECISE_MIN <= dt <= PRECISE_MAX)

def _as_heliocentric(px, py, pz, sx, sy, sz, heliocentric=True):
    """Return (x,y,z) shifted to heliocentric frame if requested."""
    if heliocentric:
        return px - sx, py - sy, pz - sz
    return px, py, pz

def _to_ecliptic(x_eq, y_eq, z_eq):
    """Rotate equatorial (ICRS) coordinates to ecliptic J2000 by +epsilon about x-axis."""
    x_e = x_eq
    y_e = _C * y_eq + _S * z_eq
    z_e = -_S * y_eq + _C * z_eq
    return x_e, y_e, z_e

def _heliocentric_ecliptic_xyz(planet, t):
    """Heliocentric ecliptic (J2000) vector(s) for a planet at time(s) t. Units: AU."""
    p = get_body_barycentric(planet, t)
    s = get_body_barycentric("sun", t)
    px = np.array(p.x.to("au").value, ndmin=1)
    py = np.array(p.y.to("au").value, ndmin=1)
    pz = np.array(p.z.to("au").value, ndmin=1)
    sx = np.array(s.x.to("au").value, ndmin=1)
    sy = np.array(s.y.to("au").value, ndmin=1)
    sz = np.array(s.z.to("au").value, ndmin=1)

    hx, hy, hz = _as_heliocentric(px, py, pz, sx, sy, sz, HELIOCENTRIC)
    ex, ey, ez = _to_ecliptic(hx, hy, hz)
    return ex, ey, ez

def _wrap_pi(angle):
    """Wrap angle (radians) into [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def _kepler_E(M, e, tol=1e-12, iters=12):
    """Solve Kepler's equation M = E - e sin E for E (radians). Vectorized."""
    M = np.array(M, dtype=float)
    E = M + e * np.sin(M)  # initial guess
    for _ in range(iters):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        dE = -f / fp
        E += dE
        if np.all(np.abs(dE) < tol):
            break
    return E

def _elements_planet(planet):
    a = SEMI_MAJOR_AU.get(planet, 1.0)
    e = ECCENTRICITY.get(planet, 0.0)
    varpi = math.radians(VARPI_DEG.get(planet, 0.0))
    L0 = math.radians(L0_DEG.get(planet, 0.0))
    P = ORBITAL_PERIOD_DAYS.get(planet, 365.256)
    n = 2 * math.pi / P  # rad/day
    M0 = _wrap_pi(L0 - varpi)
    return a, e, varpi, n, M0

def _kepler_xy_series(planet, days_since_j2000):
    """Elliptical, planar (ecliptic) heliocentric XY for an array of day offsets."""
    a, e, varpi, n, M0 = _elements_planet(planet)
    M = _wrap_pi(M0 + n * days_since_j2000)
    E = _kepler_E(M, e)
    r = a * (1 - e * np.cos(E))
    tan_v2 = np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2)
    v = 2 * np.arctan(tan_v2)
    theta = _wrap_pi(v + varpi)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def _approx_xy(planet, date):
    t = Time(date).tdb
    days = (t - J2000_TDB).to_value(u.day)
    x, y = _kepler_xy_series(planet, np.array([days]))
    return float(x[0]), float(y[0])

def planet_positions(planets, date, heliocentric=HELIOCENTRIC):
    """Return dict {planet: (x_au, y_au)} for a given ISO date or datetime/date."""
    coords = {}
    if _use_precise(date):
        t = Time(date).tdb
        for planet in planets:
            ex, ey, ez = _heliocentric_ecliptic_xyz(planet, t)
            coords[planet] = (float(ex[0]), float(ey[0]))
        return coords
    # Elliptical planar fallback
    print("[approx] Using elliptical Kepler fallback (outside precise date range)")
    for planet in planets:
        coords[planet] = _approx_xy(planet, date)
    return coords

def planet_arc(planet, date, days=ARC_DAYS, n=ARC_SAMPLES, heliocentric=HELIOCENTRIC):
    """Compute (x[], y[]) along the planet's path for ±days around date (ecliptic XY)."""
    if _use_precise(date):
        ts = Time(date).tdb + np.linspace(-days, days, n) * u.day
        ex, ey, ez = _heliocentric_ecliptic_xyz(planet, ts)
        return list(ex), list(ey)
    # Elliptical fallback
    t0 = Time(date).tdb
    days0 = (t0 - J2000_TDB).to_value(u.day)
    ds = days0 + np.linspace(-days, days, n)
    x, y = _kepler_xy_series(planet, ds)
    return list(x), list(y)

def planet_orbit_path(planet, date, samples=ORBIT_SAMPLES, heliocentric=HELIOCENTRIC):
    """Return a thin full-orbit polyline (ecliptic XY)."""
    if _use_precise(date):
        P = ORBITAL_PERIOD_DAYS.get(planet, 365.256)
        ts = Time(date).tdb + np.linspace(0.0, P, samples) * u.day
        ex, ey, ez = _heliocentric_ecliptic_xyz(planet, ts)
        return list(ex), list(ey)
    # Elliptical fallback: one full revolution
    P = ORBITAL_PERIOD_DAYS.get(planet, 365.256)
    t0 = Time(date).tdb
    days0 = (t0 - J2000_TDB).to_value(u.day)
    ds = days0 + np.linspace(0.0, P, samples)
    x, y = _kepler_xy_series(planet, ds)
    return list(x), list(y)

def planet_orbit_path_xyz(planet, date, samples=ORBIT_SAMPLES):
    """As above, but return (x,y,z) in ecliptic J2000 (heliocentric)."""
    if _use_precise(date):
        P = ORBITAL_PERIOD_DAYS.get(planet, 365.256)
        ts = Time(date).tdb + np.linspace(0.0, P, samples) * u.day
        ex, ey, ez = _heliocentric_ecliptic_xyz(planet, ts)
        return np.array(ex), np.array(ey), np.array(ez)
    # Elliptical fallback: z=0
    P = ORBITAL_PERIOD_DAYS.get(planet, 365.256)
    t0 = Time(date).tdb
    days0 = (t0 - J2000_TDB).to_value(u.day)
    ds = days0 + np.linspace(0.0, P, samples)
    x, y = _kepler_xy_series(planet, ds)
    z = np.zeros_like(x)
    return np.array(x), np.array(y), z

# --------------------------
# Figure construction
# --------------------------
def build_figure(date):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Inner Planets (Mercury–Mars)", "Outer Planets (Jupiter–Neptune)"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]]
    )

    # Inner panel
    inner_coords = planet_positions(INNER_PLANETS, date)
    for planet in INNER_PLANETS:
        # Full orbit (thin, neutral)
        ox, oy = planet_orbit_path(planet, date)
        fig.add_trace(
            go.Scatter(
                x=ox, y=oy, mode="lines",
                line=dict(width=ORBIT_LINE_WIDTH, color=ORBIT_LINE_COLOR),
                name=f"{planet} orbit"
            ),
            row=1, col=1
        )
        # ±15 d arc (thicker, planet colour)
        ax, ay = planet_arc(planet, date)
        fig.add_trace(
            go.Scatter(
                x=ax, y=ay, mode="lines",
                line=dict(width=ARC_LINE_WIDTH, color=PLANET_COLOR.get(planet, "#444")),
                name=f"{planet} ±{ARC_DAYS}d"
            ),
            row=1, col=1
        )
        # Current position (planet colour + per-planet size)
        x0, y0 = inner_coords[planet]
        fig.add_trace(
            go.Scatter(
                x=[x0], y=[y0], mode="markers+text",
                text=[planet.capitalize()], textposition="top center",
                marker=dict(
                    size=MARKER_SIZE.get(planet, 9),
                    color=PLANET_COLOR.get(planet, "#444")
                ),
                name=planet
            ),
            row=1, col=1
        )

    # Outer panel
    outer_coords = planet_positions(OUTER_PLANETS, date)
    for planet in OUTER_PLANETS:
        # Full orbit (thin, neutral)
        ox, oy = planet_orbit_path(planet, date)
        fig.add_trace(
            go.Scatter(
                x=ox, y=oy, mode="lines",
                line=dict(width=ORBIT_LINE_WIDTH, color=ORBIT_LINE_COLOR),
                name=f"{planet} orbit"
            ),
            row=1, col=2
        )
        # ±15 d arc (thicker, planet colour)
        ax, ay = planet_arc(planet, date)
        fig.add_trace(
            go.Scatter(
                x=ax, y=ay, mode="lines",
                line=dict(width=ARC_LINE_WIDTH, color=PLANET_COLOR.get(planet, "#444")),
                name=f"{planet} ±{ARC_DAYS}d"
            ),
            row=1, col=2
        )
        # Current position (planet colour + per-planet size)
        x0, y0 = outer_coords[planet]
        fig.add_trace(
            go.Scatter(
                x=[x0], y=[y0], mode="markers+text",
                text=[planet.capitalize()], textposition="top center",
                marker=dict(
                    size=MARKER_SIZE.get(planet, 9),
                    color=PLANET_COLOR.get(planet, "#444")
                ),
                name=planet
            ),
            row=1, col=2
        )

    # Sun at origin (heliocentric), yellow with thin black border in both panels
    sun_marker = dict(size=14, color="#FFD700", line=dict(color="#000000", width=1))
    fig.add_trace(
        go.Scatter(x=[0], y=[0], mode="markers+text", text=["Sun"],
                   textposition="bottom center", marker=sun_marker),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[0], y=[0], mode="markers+text", text=["Sun"],
                   textposition="bottom center", marker=sun_marker),
        row=1, col=2
    )

    # Layout and aspect
    fig.update_layout(
        title=f"Solar System Plan View — {Time(date).utc.iso.split('T')[0]}",
        showlegend=False,
        width=1100,
        height=520,
        margin=dict(l=40, r=10, t=60, b=40)
    )
    # Force equal aspect ratio in both panels and set ranges
    fig.update_xaxes(title_text="x [AU]", scaleanchor="y", scaleratio=1, constrain="domain", range=[-2, 2], row=1, col=1)
    fig.update_yaxes(title_text="y [AU]", constrain="domain", range=[-2, 2], row=1, col=1)
    fig.update_xaxes(title_text="x [AU]", scaleanchor="y2", scaleratio=1, constrain="domain", range=[-35, 35], row=1, col=2)
    fig.update_yaxes(title_text="y [AU]", constrain="domain", range=[-35, 35], row=1, col=2)

    return fig

# --------------------------
# Rendering helpers (avoid fig.show to bypass nbformat dependency)
# --------------------------
def save_figure_html(fig, output_path=OUTPUT_HTML, include_js="cdn"):
    """Save a standalone interactive HTML file and return its path."""
    html = fig.to_html(full_html=True, include_plotlyjs=include_js)
    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
    return str(output_path)

def render_figure(fig):
    """Display the figure without using plotly.io.show().
    If ipywidgets/IPython are available, embed HTML inline; otherwise save to file.
    """
    try:
        html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        display(HTML(html))
    except Exception:
        path = save_figure_html(fig)
        print(f"Saved interactive HTML to '{path}'. Open this file in a browser.")

# --------------------------
# Tests
# --------------------------
def _is_finite_pair(p):
    return all(np.isfinite([p[0], p[1]]))

def run_smoke_tests():
    test_dates = [
        "2025-08-24",  # current default
        "2000-01-01",  # Y2K sanity
        "1990-06-15",  # historical
        "1987-07-10",  # additional coverage
        "1800-01-01",  # force approx fallback (past)
        "2200-01-01",  # force approx fallback (future)
    ]

    for td in test_dates:
        inner = planet_positions(INNER_PLANETS, td)
        outer = planet_positions(OUTER_PLANETS, td)
        # Keys present
        assert set(inner.keys()) == set(INNER_PLANETS)
        assert set(outer.keys()) == set(OUTER_PLANETS)
        # Finite numbers
        assert all(_is_finite_pair(inner[p]) for p in INNER_PLANETS)
        assert all(_is_finite_pair(outer[p]) for p in OUTER_PLANETS)
        # Arc lengths and finiteness
        ax, ay = planet_arc("earth", td, days=ARC_DAYS, n=ARC_SAMPLES)
        assert len(ax) == ARC_SAMPLES and len(ay) == ARC_SAMPLES
        assert all(np.isfinite(ax)) and all(np.isfinite(ay))
        # Orbit finiteness for representatives
        for pl in ("earth", "jupiter"):
            ox, oy = planet_orbit_path(pl, td, samples=200)
            assert len(ox) == 200 and len(oy) == 200
            assert all(np.isfinite(ox)) and all(np.isfinite(oy))

    # Figure has the expected number of traces: 3 per planet * 8 + 2 Sun markers = 26
    fig_test = build_figure("2025-08-24")
    assert len(fig_test.data) == 26

    # Date parser fallbacks
    assert _parse_date("2025-08-24") == "2025-08-24"
    assert _parse_date("bad-date", fallback="1999-12-31") == "1999-12-31"

    # Heliocentric distance sanity checks (3D magnitude, broad bounds)
    ex, ey, ez = planet_orbit_path_xyz("earth", "2025-08-24", samples=600)
    er = np.sqrt(ex**2 + ey**2 + ez**2)
    assert 0.97 < er.min() < 1.00 and 1.00 < er.max() < 1.03

    mx, my, mz = planet_orbit_path_xyz("mercury", "2025-08-24", samples=600)
    mr = np.sqrt(mx**2 + my**2 + mz**2)
    assert 0.25 < mr.min() < 0.40 and 0.40 < mr.max() < 0.60

    rx, ry, rz = planet_orbit_path_xyz("mars", "2025-08-24", samples=600)
    rr = np.sqrt(rx**2 + ry**2 + rz**2)
    assert 1.30 < rr.min() < 1.50 and 1.60 < rr.max() < 1.75

    # Extra checks specifically in fallback mode
    for td in ("1800-01-01", "2200-01-01"):
        ex, ey = planet_orbit_path("earth", td, samples=720)
        er = np.hypot(ex, ey)
        assert 0.97 < float(np.min(er)) < 1.00 and 1.00 < float(np.max(er)) < 1.03

    print("Smoke tests passed for:", ", ".join(test_dates))

# --------------------------
# UI glue (date entry box) & run
# --------------------------
def _parse_date(val, fallback="2025-08-24"):
    if isinstance(val, (datetime,)):
        return val.date().isoformat()
    if isinstance(val, (date_cls,)):
        return val.isoformat()
    if isinstance(val, str):
        s = val.strip()
        try:
            datetime.strptime(s, "%Y-%m-%d")
            return s
        except Exception:
            return fallback
    return fallback

if __name__ == "__main__":
    date_str = "2025-08-24"

    # Run tests first
    run_smoke_tests()

    # Build initial figure
    fig = build_figure(date_str)

    if HAVE_WIDGETS:
        try:
            default_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            default_dt = datetime.utcnow().date()

        dp = widgets.DatePicker(description="Date", value=default_dt)
        btn = widgets.Button(description="Update")
        info = widgets.HTML(value=f"<em>Full orbits + arcs (±{ARC_DAYS} days)</em>")
        ui = widgets.HBox([dp, btn, info])

        def _on_click(_):
            sel = _parse_date(dp.value or date_str, fallback=date_str)
            clear_output(wait=True)
            display(ui)
            render_figure(build_figure(sel))

        btn.on_click(_on_click)
        display(ui)

    # Render without relying on plotly's nbformat-dependent show()
    render_figure(fig)

    # Also save a standalone HTML for use outside notebooks
    path = save_figure_html(fig, OUTPUT_HTML)
    print(f"Standalone interactive file saved to: {path}")
