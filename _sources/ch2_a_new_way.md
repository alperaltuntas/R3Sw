# Chapter 2: Reasoning About Code - Abstraction and Specification

In Chapter 1, we implemented a working 1-D heat equation solver.
It runs, but it’s fragile, opaque, and hard to test or extend.

This resembles a common situation in scientific computing:

 - write some code, see if it runs, fix what breaks, repeat. 
 - It gets results quickly, but the codebase becomes harder to understand and maintain over time.

In this chapter, we ask: can we design our code systematically, in a way that allows us to reason about it before we even run it?

To achieve this, we need to manage complexity by breaking the problem into smaller pieces.

But how do we decide how to break a problem into pieces?
 - We can use abstraction to identify the key concepts in our problem, and then use those abstractions to guide our decomposition.
 - Once we have a decomposition, we can write specifications for each piece, i.e., precise and testable descriptions of what each piece should do.

The process of designing software in this way can be summarized in four steps:

 1. **Abstraction:** determine the concepts that matter.
 2. **Decomposition:** Based on our abstractions, break the problem into smaller pieces.
 3. **Specification:** For each piece, write down what it should do, in a way that’s precise and testable.
 4. **Implementation:** Write code that meets the specifications.

*Note: This kind of design process is common in traditional software engineering, but it’s not often followed in scientific computing. The pioneers and advocates of this and similar approaches include [Barbara Liskov](https://en.wikipedia.org/wiki/Barbara_Liskov),...*


### 1. Abstraction

In software design, **abstraction** is the first and most important step. Here, we form a high-level architecture for our system:
 - determine high-level concepts that matter.
 - ignore irrelevant details.

This is a creative step, where we explore different ways to represent the system and its components.

In scientific computing, this step isn’t given enough attention, but it is crucial:

> What matters is the fundamental structure of the design. If you get
> it wrong, there is no amount of bug fixing and refactoring that will
> produce a reliable, maintainable, and usable system
> *- D. Jackson. The essence of software. (2021)*

As scientists and engineers, we are already accustomed to abstraction: we model complex systems by removing unnecessary details and breaking problems into components.


![Abstraction](./img/esm.png)



**A natural concern:** Aren’t most bugs caused by subtle low-level issues?

Yes, but many “low-level” bugs are symptoms of unclear high-level intent. When top-level responsibilities are muddled, they trickle down into brittle, error-prone implementations.
Abstraction pays off by:
 -  clarifying intent
 - making low-level details easier to manage.


#### Data Abstraction vs. Procedural Abstraction

In software design, there are two main forms of abstraction:

- **Data Abstraction**: defining data structures that encapsulate state and behavior.
- **Procedural Abstraction**: defining procedures (functions, methods) that encapsulate behavior.

For our running example, we will use procedural abstraction, since that’s simpler and more appropriate for a small solver like the 1-D heat equation.

For larger and more sophisticated applications, data abstraction (e.g., object-oriented programming) may be more appropriate. However, the fundamental principles of reasoning apply equally well to both.


### 2. Decomposition

If abstraction helps us decide what matters, decomposition helps us decide how to organize it.

We want decomposition to:

 - Isolate concerns and manage complexity

 - Facilitate testing and reasoning

 - Improve maintainability and extensibility


Forms of decomposition include functions/methods (procedures), classes, and modules.

In the case of procedural abstraction, guiding principles in breaking a computation into procedures are:

 - Each procedure should have a single, well-defined purpose.
 - Procedures should be general: avoid hard-coding specific values or assumptions.
 - Each procedure should be as simple as possible, but no simpler:
    > “A good check for simplicity is to give the procedure a name that describes its purpose.
    >  If it is difficult to think of a name, there may be a problem with the procedure.” -Liskov


### 3. Specification: Writing Down Intent

Once we have pieces, we need to say what each one should do.

A *specification* is a precise statement of what software should do, independent of how it does it.

Specifications can range from informal descriptions in plain English
to fully formal mathematical models.

In this tutorial, we’ll use a **lightweight, practical level** of specification that combines:
 - **type annotations** (for inputs and outputs),
 - **assertions** that serve as contracts, i.e., preconditions and postconditions.


This lightweight approach:
 - Guides implementation,
 - Facilitates testing and validation.
 - Embeds reasoning directly into the program.

### A simple example:

Below simple function exemplifies our practical approach to specification:
 - Type annotations specify that `div` takes two floats and returns a float.
 - The first `assert` is a precondition (P): it specifies what must be true before the function is called.
 - The second `assert` is a postcondition (Q): it specifies what must be true after the function is called.


```python
def div(x: float, y: float) -> float: # Type annotations for inputs and output
    assert y != 0           # P    (precondition)
    res = x / y             # code (implementation)
    assert res * y == x     # Q    (postcondition)
    return res
```


```python
div(4.0, 2.0)
```




    2.0



**Note:** Historically, preconditions and postconditions are often denoted `P` and `Q`, respectively, as in *Hoare logic*: 

    {P} C {Q}
where C is a command (procedure).


**Note:** Executable assertions as specifications is not universally loved. 
Some prefer to keep specifications as comments or separate documents. 
However, embedding specifications as executable assertions enables runtime checking and testing, 
something we'll utilize in the remaining chapters. Assertions can be disabled in production code,
e.g., via `-O` flag in Python, if performance is critical.


### Exercise 2.1:

Can you find inputs to `f` where this function fails the postcondition even when the precondition is satisfied?



```python
div(_, _)
```




    1.0



**Answer:** One possible input is `f(7, 25)`. The precondition `y != 0` is satisfied, but the postcondition `res * y == x` fails because due to floating-point rounding error.


```python
div(7, 25)
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    Cell In[4], line 1
    ----> 1 div(7, 25)


    Cell In[1], line 4, in div(x, y)
          2 assert y != 0           # P    (precondition)
          3 res = x / y             # code (implementation)
    ----> 4 assert res * y == x     # Q    (postcondition)
          5 return res


    AssertionError: 


**Note:** Floating-point arithmetic introduces unavoidable rounding errors. 
These aren’t bugs, but natural computational artifacts. We can *weaken* the postcondition
to account for this by using approximate equality. We can do so using `pytest.approx` (or, similarly, `math.isclose`, `numpy.isclose`, etc.):


```python
from pytest import approx

def div(x: float, y: float) -> float: # Type annotations for inputs and output
    assert y != 0               # P    (precondition)
    res = x / y                 # code (implementation)
    assert res * y == approx(x) # Q    (postcondition)
    return res
```


```python
div(7, 25)
```




    0.28



**Takeaway**: Postconditions should reflect computational realities (e.g., floating-point).

**Preconditions deserve similar care.**

 **weakest precondition** :
    the *most general condition* under which the code
    works correctly. It ensures your function is usable in the widest range of scenarios
    while still preventing incorrect behavior.

Conversely, we want the **strongest postcondition** possible: the most precise guarantee
about the outcome. This balance maximizes both the flexibility and reliability of our code.



### 4. Implementation: Writing Code that Meets Specifications

With abstractions identified, pieces decomposed, and specifications written, we can implement code that satisfies them.

This shifts our mindset: instead of starting with lines of code and hoping they work, we start with clear expectations and then fill in implementations that must meet them.

We can then use testing and verification to ensure our implementations meet their specifications instantly.

## Heat Equation Solver, Revisited

In Chapter 1, we wrote a quick, monolithic solver for the 1-D heat equation.
It worked, but it mixed boundary conditions, flux calculations, and time-stepping into a single opaque function.

Now we’ll redesign it systematically, using the four-step process:

Abstraction → Decomposition → Specification → Implementation.

### Step 1 — Abstraction

At this stage, we define the *what* of our problem, i.e., the key concepts we need to model:

 - We want to advance a 1-D temperature field $u_i$ on a uniform grid using a conservative finite volume method:

$
\qquad
u_i^{n+1} = u_i^{n} + \Delta t \times \, div(F_i),
\qquad
div(F_i) = \frac{F_{i} - F_{i+1}}{\Delta x},
\qquad
F_i =
\begin{cases}
q_L & i = 0, \\[6pt]
-\kappa \dfrac{u_i - u_{i-1}}{\Delta x} & 1 \leq i \leq N-1, \\[10pt]
q_R & i = N.
\end{cases}
$

The key concepts we need are:

 - Mesh: describes the geometry of the domain (dx, number of cells).
 - Cell fields (u, dudt): quantities stored at cell centers.
 - Face fields (F): quantities stored at cell faces.
 - Procedures to perform the time-stepping of the heat equation.

These abstractions are all we need to model the system.


### Step 2 – Decomposition

Based on our abstractions, we can break the solver into smaller pieces. Since we'll use procedural abstraction, our focus is on defining functions that carry out the computations. But first, let's determine the key data structures that we'll need:

 #### Data Structures:

   - *mesh*: discrete representation of the computational domain (grid spacing, number of cells).
   - *vector*: a simple 1-D array of floats.
   - *cell field*: a vector storing cell-centered values (e.g., temperature, tendencies).
   - *face field*: a vector storing face-centered values (e.g., diffusive fluxes).

Let's start simple and define a type alias for vectors:


```python
vec = list[float]
```


Next, let's define a mesh *data class*, i.e., a simple data structure intended for storing data without much additional functionality. Since our approach is procedural, we won't rely much on object-oriented programming features beyond this simple data class.


```python
from dataclasses import dataclass

@dataclass
class Mesh:
    """Uniform 1-D mesh."""
    
    dx: float  # cell size
    N: int     # number of cells

    def cell_field(self) -> vec:
        return [0.0] * self.N

    def face_field(self) -> vec:
        return [0.0] * (self.N + 1)
```

Notice the helper methods `cell_field` and `face_field` that create zero-initialized fields of the appropriate size based on the mesh.


#### Procedures:

Recall the abstraction and decomposition principles we discussed for procedural abstraction, i.e.,
 - Each procedure should have a single, well-defined purpose.
 - Procedures should be general: avoid hard-coding specific values or assumptions.
 - Each procedure should be as simple as possible, but no simpler.

Based on these principles, we can define the following *general* procedures:

 - `apply_bc(f, bc)`: apply boundary conditions to face fluxes.
 - `diffusive_flux(f, c, kappa, dx)`: compute interior fluxes.
 - `divergence(c, f, dx)`: compute tendencies from fluxes.

The argument names use conventions to clarify the placement of vectors on the mesh:

 - `f` to denote face fields.
 - `c` to denote cell fields.

This keeps the procedures general and reusable for different types of fields. If we were to compute the diffusive fluxes of a different quantity (e.g., concentration), we could reuse the same `diffusive_flux` procedure without modification. (This is called *abstraction by parameterization*.)

In addition to the above general procedures, we need two more, specific to the heat equation solver. For
these, the argument names are more specific to clarify their purpose:

  - `step_heat_eqn(u, F, dudt, kappa, dt, mesh, bc)`: advance one time step.
  - `solve_heat_eqn(u0, kappa, dt, nt, dx, bc)`: orchestrate multiple steps.

### Step 3 – Specification

Recall, our practical approach to specification combines:
 - type annotations
 - pre- and postconditions as assertions.

First, let's write a core API that capture our procedural abstractions. This will
include only the function signatures, i.e., functions names, arguments, and brief descriptions.
While doing so, we'll include the first constituent of our specifications: type annotations.

This isn’t about writing code yet: it’s about specifying the conceptual architecture of the system.


```python
# Procedures:

def apply_bc(f: vec, bc: vec) -> None:
    """Apply BCs by overriding first and last face quantities (f)."""
    ...

def diffusive_flux(f: vec, c: vec, kappa: float, dx: float) -> None:
    """Given a cell field (c), compute the diffusive flux (f)."""
    ...

def divergence(c: vec, f: vec, dx: float) -> None:
    """Compute the divergence of face quantities (f) and store in (c)."""
    ...

def step_heat_eqn(u: vec, F: vec, dudt: vec, kappa: float, dt: float, mesh: Mesh, bc: vec) -> list:
    """Advance cell field u by one time step using explicit Euler method."""
    ...

def solve_heat_eqn(u0: vec, kappa: float, dt: float, nt: int, dx: float, bc: vec) -> vec:
    """Orchestrate nt steps over cell field u."""
    ...

```

Having defined the core API and type annotations, we can now add the next layer of specifications: 

 - preconditions and postconditions as executable assertions.

Take the `divergence` function that computes the divergence of face fluxes:

$\qquad
\nabla \cdot F = \frac{F_{i} - F_{i+1}}{\Delta x}
\qquad$

Given the above formula, one can identify two key preconditions:
 - The vector `c` (output) and the vector `f` (input) must have compatible sizes: specifically, `len(c) == len(f) - 1`.
 - The mesh spacing `dx` must be positive.


```python
def divergence(c: vec, f: vec, dx: float) -> None:
    """Compute the divergence of face quantities (f) and store in (c)."""
    assert len(c) == len(f) - 1, "Size mismatch"
    assert dx > 0, "Non-positive dx"
    ...
```

As for the postcondition, we can use a mathematical property of the divergence operator to define it precisely:

**Telescoping Property of Divergence**: The sum of the divergence over all cells equals the net flux through the boundaries.

$\qquad
\sum_{i=0}^{N-1} (\nabla \cdot F)_i = F_0 - F_N
\qquad$

We can encode this property as a function that returns `True` if the property holds, and `False` otherwise:


```python
def telescoping(c: vec, f: vec, dx: float) -> bool:
    """Check the finite volume telescoping property."""
    total_divergence = sum(c) * dx
    boundary_flux = f[0] - f[-1]
    return total_divergence == approx(boundary_flux)
```

We can then add this check as a postcondition in the `divergence` function:



```python
def divergence(c: vec, f: vec, dx: float) -> None:
    """Compute the divergence of face quantities (f) and store in (c)."""
    assert len(c) == len(f) - 1, "Size mismatch"
    assert dx > 0, "Non-positive dx"
    ...
    assert telescoping(c, f, dx)           # {C} Postcondition
```

### Exercise 2.2:

One of the key preconditions for the Euler (FTCS) scheme is the stability condition as given below:

$\qquad
\Delta t \leq \frac{\Delta x^2}{2 \kappa}
\qquad$

Implement a function `stability_condition(...)` that checks this condition. Which 
function(s) would you add this as a precondition to?



```python
def stability_condition(dt: float, dx: float, kappa: float) -> bool:
    """Check the stability condition for the explicit Euler scheme."""
    ...
```

**Answer:**

Below is a possible implementation of the `is_stable` function:


```python
def stability_condition(dt: float, dx: float, kappa: float) -> bool:
    """Check the stability condition for the explicit Euler scheme."""
    return dt <= (dx ** 2) / (2 * kappa)
```

The function `stability_condition` can be added as a precondition to the `solve_heat_eqn` function, as it ensures that the time step `dt` is appropriate for the spatial discretization `dx` and the diffusion coefficient `kappa`, which is crucial for the stability of the explicit Euler scheme used in the heat equation solver.

### Exercise 2.3:

Now let's check whether heat is conserved in our solver. The condition for conservation can be expressed as:

$$
\sum_{i=0}^{N-1} u_i^{n+1} \Delta x = \sum_{i=0}^{N-1} u_i^{n} \Delta x + \Delta t \left( q_L - q_R \right).
$$

with special Cases:

 - If `q_L` == `q_R` == 0, the mean stays constant.
 - If `q_L` and `q_R` are constant, the temperature profile converges to a linear steady state.


```python
def heat_is_conserved(u_sum_old: float, u_sum_new: float, dt: float, F: vec) -> bool:
    """Check if heat is conserved."""
    ...
```

**Answer:**

The he following cell contains a possible implementation of the heat conservation check. As for which function(s) would need to be modified to incorporate this check, it may be added as a postcondition to the step function, or, alternatively, it may be added as a loop invariant in the timestepping loop within the solve function.


```python
def heat_is_conserved(u_sum_old: float, u_sum_new: float, dt: float, F: vec) -> bool:
    """Check if heat is conserved."""
    expected_change = dt * (F[0] - F[-1])
    actual_change = u_sum_new - u_sum_old
    return actual_change == approx(expected_change)
```


**Loop invariants**

A loop invariant is a condition (a boolean expression) that holds true before and after each iteration of a loop. It helps reason about the correctness of the loop by ensuring that certain properties hold:

 - before the loop starts
 - after each iteration of the loop

which guarantees that once the loop has finished executing, the desired property holds for the final state.

In the case of the heat equation solver, ensuring that heat is conserved after each time step guarantees that the total heat within the domain remains constant.



### Step 4 – Implementation

With abstractions identified, pieces decomposed, and specifications written, we can implement code that satisfies them.


```python
def apply_bc(f: vec, bc: vec) -> None:
    """Apply BCs by overriding first and last face quantities (f)."""
    assert len(f) > 1
    f[0], f[-1] = bc[0], bc[1]

def diffusive_flux(f: vec, c: vec, kappa: float, dx: float) -> None:
    """Given a cell field (c), compute the diffusive flux (f)."""
    assert len(f) == len(c) + 1, "Size mismatch"
    assert dx > 0, "Non-positive dx"
    for i in range(1, len(f) - 1):
        f[i] = -kappa * (c[i] - c[i-1]) / dx

def divergence(c: vec, f: vec, dx: float) -> None:
    """Compute the divergence of face quantities (f) and store in (c)."""
    assert len(c) == len(f) - 1, "Size mismatch"
    assert dx > 0, "Non-positive dx"
    for i in range(len(c)):
        c[i] = (f[i] - f[i+1]) / dx
    assert telescoping(c, f, dx)

def step_heat_eqn(u: vec, F: vec, dudt: vec, kappa: float, dt: float, mesh: Mesh, bc: vec) -> list:
    """Advance cell field u by one time step using explicit Euler method."""
    assert dt > 0, "Non-positive dt"
    assert mesh.N == len(u) == len(dudt) == len(F) - 1, "Size mismatch"

    apply_bc(F, bc)
    diffusive_flux(F, u, kappa, mesh.dx)
    divergence(dudt, F, mesh.dx)

    for i in range(mesh.N):
        u[i] += dt * dudt[i]

def solve_heat_eqn(u0: vec, kappa: float, dt: float, nt: int, dx: float, bc: vec) -> vec:
    """Orchestrate nt steps over cell field u."""

    assert nt > 0, "Number of time steps must be positive"
    assert len(bc) == 2, "Boundary conditions must be a list of two values"
    assert stability_condition(dt, dx, kappa), "Stability condition not met"

    mesh = Mesh(dx, N=len(u0))
    u = u0.copy()
    F = mesh.face_field()
    dudt = mesh.cell_field()

    for _ in range(nt):
        u_sum_old = sum(u) * mesh.dx
        step_heat_eqn(u, F, dudt, kappa, dt, mesh, bc)
        u_sum_new = sum(u) * mesh.dx
        assert heat_is_conserved(u_sum_old, u_sum_new, dt, F), "Heat not conserved"

    return u

```


```python
solve_heat_eqn(
    u0 = [0.0, 100.0, 0.0],
    kappa = 0.1,
    dt = 1.0,
    nt = 1000,
    dx = 1.0,
    bc = [0.0, 0.0]
)
```




    [33.333333333333314, 33.33333333333333, 33.333333333333314]



## What we just did

We now have a solver whose structure and reasoning are explicit. It’s modular: each function 
has a single purpose over well-defined data.

We did write a lot more lines, because we added:
 - Isolation of concerns (apply_bc, flux, divergence, step, solve) instead of one monolith
 - Type annotations
 - Executable contracts (pre/postconditions, invariants)
 - Lightweight data types (Mesh, vec)

**In real applications**, this "ceremony" is a small fraction of total code: The heavy lifting
lives in numerical computations, I/O, parallelization, physics, etc.. The scaffolding pays
off by reducing change amplification and cognitive load:

 - Safer changes (localized edits)
 - Better testing (unit, property, etc.)
 - Faster debugging (failures point to where)
 - Reusability (shared abstractions)

**Do assertions hurt performance?** Potentially, if run every step. In practice you can disable
them for production (`python -O`) or sample them periodically during long runs.

## Looking Ahead

In **Chapter 3**, we’ll take the next step:  
turning these specifications into unit tests that run automatically,
giving us rapid feedback and confidence as our code evolves.


---

*This notebook is a part of the "Rigor and Reasoning in Research Software" (R3Sw) tutorial, led by Alper Altuntas and sponsored by the Better Scientific Software (BSSw) Fellowship Program. Copyright © 2025*


