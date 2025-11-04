[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/alperaltuntas/r3sw/HEAD)

# Welcome

Welcome to the Rigor and Reasoning in Research Software (R3Sw) Tutorial!

Over the course of this tutorial, we’ll explore how to make our scientific software more reliable, understandable, and trustworthy. And we'll aim to do so without losing the creativity and speed that make research exciting.

## Launching notebooks

[Click here](https://mybinder.org/v2/gh/alperaltuntas/r3sw/HEAD) to launch a Jupyter Hub online via binder to interactively explore the notebooks.

## Local installation

Download or clone this repository to your local machine:

```bash
git clone https://github.com/alperaltuntas/r3sw.git
cd r3sw
```

Then, create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate r3sw
```

You can then launch Jupyter Notebook:

```bash
cd notebooks
jupyter-lab
```


## Why This Matters

Scientific software is at the heart of modern research.
It powers everything from climate simulations to molecular modeling to AI-driven discoveries.

But all too often, software is built in a hurry, using a "code-and-fix" style:
 - we write code quickly to get results, fix issues as they come up, and hope it holds together.
 - The result? Code that works... most of the time, but is fragile, hard to test, and tricky to extend when the next research question comes along.

This tutorial is about breaking that cycle.
 - We’ll take inspiration from the scientific method itself. Just as good science relies on forming hypotheses, testing them, and refining ideas, good scientific software can be built using that same process.
 - By applying rigor and reasoning to our code, we can create tools that are more robust, easier to work with, and better aligned with the science they support.

## What We’ll Do

We’ll work through a running example: a simple 1-D heat equation solver.

We’ll start with an ad-hoc, monolithic prototype, just like the quick scripts many of us have written. And gradually transform it into modular, testable, and trustworthy software.

Along the way, you’ll get hands-on experience with techniques like:

 - **Designing for robustness:** separation of concerns, specifications, preconditions, postconditions, and invariants

 - **Unit testing with pytest:** the essential foundation for building confidence in code

 - **Property-based testing with Hypothesis:** exploring edge cases automatically

 - **Theorem proving with Z3:** exhaustively reasoning about code behavior

By the end, you’ll have a set of tools and habits you can bring back to your own projects.

## Target Audience

This tutorial is for scientists, engineers, and students working in scientific computing across any domain and at any career stage.

 - You don’t need prior experience with testing or verification.

 - Familiarity with Python is necessary, but that’s all you need.

Whether you’re just getting started or have years of experience, we hope you’ll find ways to level up your skills and make your software more trustworthy.

## The Ladder of Rigor

Throughout the tutorial, we’ll climb what we call a *ladder of rigor.*
Each step represents a different way to build confidence in your code:

- **Unit testing:** the foundation: simple and practical, but limited in coverage.

- **Property-based testing:** expands coverage and catches surprising edge cases.

- **Theorem proving:** the most rigorous, offering deep guarantees in specialized cases.

Each step adds new power and comes with its own tradeoffs.
You’ll learn how to choose the right approach for your project’s needs.

## A Shared Goal

More than just learning tools, this tutorial is about adopting a scientific mindset for software:

 - Think about what properties your code should satisfy.
 - Make those expectations explicit through specifications and reasoning.
 - Test and challenge those expectations, just as you would test a scientific hypothesis.
 - Refine and improve both your code and your understanding.

By approaching software this way, you’re not just writing code: You’re conducting an ongoing experiment to make your science more reliable and reproducible.

Let’s get started!

### Tutorial Lead

 - Alper Altuntas (NSF NCAR)

### Guest Lecturers and Invited Speakers

 - Soonho Kong (AWS)
 - Adrianna Foster (NSF NCAR)
 - Antonios Mamalakis (UVA)
 - Deepak Cherian (Earthmover)
 - Helen Kershaw (NSF NCAR)
 - Manish Venumuddula (NSF NCAR)

### Program Committee 

 - John Baugh (NCSU)
 - Ilene Carpenter (HPE)
 - Brian Dobbins (NSF NCAR)
 - Michael Duda (NSF NCAR)
 - Karsten Peters-von Gehlen (DKRZ)
 - Ganesh Gopalakrishnan (Utah)
 - Dorit Hammerling (Mines)
 - Balwinder Singh (PNNL)
