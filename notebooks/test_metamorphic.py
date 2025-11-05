# Run these tests on the command-line with
#     pytest --hypothesis-verbosity=verbose -s test_metamorphic.py
# to see what cases hypothesis tries.
#
# Possible improvements:
# 1. Users are most likely to run the solver near the stability boundary.
#    We should adjust the `meshes` strategy to generate many more cases near that boundary
# 2. We could create a new strategy `systems` that return instances of a new dataclass
#    that encapsulated everything needed to integrate. This would necessarily take a strategy
#    to generate initial conditions as input.


import math
import pytest
from hypothesis import assume, given
import hypothesis.strategies as st
from heat1d import Mesh, step_heat_eqn, vec

floats_st = st.floats(
    min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
)
kappa_st = st.floats(
    min_value=1e-4, max_value=10, allow_nan=False, allow_infinity=False
)
dx_st = st.floats(min_value=1e-3, max_value=10, allow_nan=False, allow_infinity=False)
dt_st = st.floats(min_value=1e-6, max_value=10, allow_nan=False, allow_infinity=False)
# Mesh sizes
n_st = st.integers(min_value=3, max_value=40)
fluxes_st = st.floats(
    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
)
bc_st = st.tuples(fluxes_st, fluxes_st)


def initial_conditions(*, min_n: int, max_n: int) -> vec:
    return st.lists(floats_st, min_size=min_n, max_size=max_n)


def symmetric_field(min_n=3, max_n=31):
    # Produce palindromic arrays
    # (odd length preferred for strict central symmetry)
    # Build half then mirror
    half = st.lists(floats_st, min_size=(min_n // 2), max_size=(max_n // 2))
    center = floats_st
    return st.builds(lambda h, c: h + [c] + list(reversed(h)), half, center)


def dxs(*, min_value=1e-3, max_value=10) -> float:
    return st.floats(
        min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False
    )


def dts(*, min_value=1e-6, max_value=10) -> float:
    return st.floats(
        min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False
    )


@st.composite
def meshes(draw: st.DrawFn, *, kappas=kappa_st, dts=dts, dxs=dxs, ns=n_st):
    # `draw` is a function that lets you draw a concrete value
    # from a strategy like so:
    kappa = draw(kappas)

    MIN_DT = 1e-6

    # now we pick a dx, dt such that the stability condition is *always* satisfied
    # 10% fudge factor to account for floating point resolution near MIN_DT
    dx = draw(dxs(min_value=1 * math.sqrt(2.1 * kappa * MIN_DT)))
    dt = draw(dts(min_value=MIN_DT, max_value=0.49 * dx**2 / kappa))

    # assert the postcondition
    # (always test your strategies!)
    assert (kappa * dt / (dx**2)) < 0.5, kappa * dt / (dx**2)

    # the need to return multiple values here suggests
    # this approach isn't a good for for this particular problem.
    return kappa, dt, Mesh(dx=dx, N=draw(ns))


@given(
    # needed to draw initial conditions
    data=st.data(),
    # draw odd-number meshes only;
    # this is where writing composite strategies with strategies as arguments is useful
    kappa_dt_mesh=meshes(ns=n_st.map(lambda n: 2 * n - 1)),
    bc=bc_st,
)
def test_symmetry_about_center(data, kappa_dt_mesh, bc):
    """
    Property:
      solution (a): bc0 ⇒| | | | | | ⇒ bc1
      solution (b): bc1 ⇐| | | | | | ⇐ bc0
    are mirror images.
    """
    kappa, dt, mesh = kappa_dt_mesh
    assert mesh.N >= 3
    assert kappa * dt / (mesh.dx**2) < 0.5  # stability condition

    u0 = data.draw(symmetric_field(min_n=mesh.N, max_n=mesh.N))
    u = u0.copy()
    for _ in range(10):
        step_heat_eqn(u, kappa, dt, mesh, bc)

    urev = u0.copy()
    # reverse the boundary conditions and swap
    bcrev = [-bc[1], -bc[0]]
    for _ in range(10):
        step_heat_eqn(urev, kappa, dt, mesh, bcrev)

    assert all(a == b for a, b in zip(u, urev[::-1], strict=True))


@given(data=st.data(), kappa_dt_mesh=meshes(), intercept=floats_st, bc=bc_st)
def test_constant_addition_invariance(data, kappa_dt_mesh, intercept, bc):
    kappa, dt, mesh = kappa_dt_mesh
    u0 = data.draw(initial_conditions(min_n=mesh.N, max_n=mesh.N))

    perturbed = [intercept + u for i, u in enumerate(u0)]

    orig = u0.copy()
    for _ in range(10):
        step_heat_eqn(orig, kappa, dt, mesh, bc)

    for _ in range(10):
        step_heat_eqn(perturbed, kappa, dt, mesh, bc)

    rescaled = [b - intercept for i, b in enumerate(perturbed)]

    assert orig == pytest.approx(rescaled)
