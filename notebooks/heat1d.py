
from dataclasses import dataclass
from pytest import approx

vec = list[float]

@dataclass
class Mesh:
    """Uniform 1-D mesh."""

    dx: float  # cell size
    N: int     # number of cells

    def cell_field(self) -> vec:
        return [0.0] * self.N

    def face_field(self) -> vec:
        return [0.0] * (self.N + 1)

def apply_bc(f_out: vec, bc: vec) -> None:
    """Apply BCs by overriding first and last face quantities (f_out)."""
    assert len(f_out) > 1, "face field size too small"
    assert len(bc) == 2, "bc must be of size 2"
    f_out[0], f_out[-1] = bc[0], bc[1]

def diffusive_flux(f_out: vec, c: vec, kappa: float, dx: float) -> None:
    """Given a cell field (c), compute the diffusive flux (f_out)."""
    assert len(f_out) == len(c) + 1, "Size mismatch"
    assert dx > 0 and kappa > 0, "Non-positive dx or kappa"
    for i in range(1, len(f_out) - 1):
        f_out[i] = -kappa * (c[i] - c[i-1]) / dx

def divergence(c_out: vec, f: vec, dx: float) -> None:
    """Compute the divergence of face quantities (f) and store in (c_out)."""
    assert len(c_out) == len(f) - 1, "Size mismatch"
    assert dx > 0, "Non-positive dx"
    for i in range(len(c_out)):
        c_out[i] = (f[i] - f[i+1]) / dx

def step_heat_eqn(F: vec, u_inout: vec, kappa: float, dt: float, mesh: Mesh, bc: vec):
    """Advance cell field u by one time step using explicit Euler method."""
    assert dt > 0, "Non-positive dt"
    assert mesh.N == len(u_inout), "Size mismatch"
    assert mesh.N == len(F), "Size mismatch in main field"

    divF = mesh.cell_field()

    diffusive_flux(F, u_inout, kappa, mesh.dx)
    divergence(divF, F, mesh.dx)

    for i in range(mesh.N):
        u_inout[i] += dt * divF[i]

def solve_heat_eqn(u0: vec, kappa: float, dt: float, nt: int, dx: float, bc: vec) -> vec:
    """Orchestrate nt steps over cell field u."""

    assert nt > 0, "Number of time steps must be positive"
    assert dt <= (dx ** 2) / (2 * kappa), "Stability condition not met"

    mesh = Mesh(dx, N=len(u0))

    # Setup face field and apply boundary conditions
    F = mesh.face_field()
    apply_bc(F, bc)

    # Solver loop
    u = u0.copy()
    for _ in range(nt):
        step_heat_eqn(F, u, kappa, dt, mesh, bc)

    return u
