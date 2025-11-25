# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MacroProg is a Julia package for solving macroeconomic models, specifically implementing Hansen's RBC (Real Business Cycle) model with stochastic growth. The project provides tools for discretizing AR(1) processes using Tauchen's method, solving dynamic programming problems using value function iteration, and conducting simulation/counterfactual analysis.

## Architecture

### Core Module Structure
- `src/MacroProg.jl`: Main module file that includes all submodules and exports key functions
- `src/AbstractTypes.jl`: Defines abstract base types for models, processes, parameters, and results
- `src/tauchen.jl`: Implementation of Tauchen's method for discretizing AR(1) processes
- `src/HansenRBC.jl`: Complete implementation of Hansen's RBC model with value function iteration

### Key Type System
The codebase uses a well-structured type hierarchy:

#### Core Abstract Types
- `AbstractModel`: Base type for economic models
- `AbstractProcess`: Base type for stochastic processes
- `AbstractParams`: Base type for model parameters
- `AbstractVariables`: Base type for endogenous variables
- `AbstractSolvingResult`: Base type for solution results

#### Simulation Abstract Types (NEW)
- `AbstractSteadyState`: Base type for steady state computations
- `AbstractAggregates`: Base type for aggregate variable containers
- `AbstractShockSeries`: Base type for realized shock series
- `AbstractSimulationResult`: Base type for simulation/counterfactual results

### Main Components

#### HansenRBC Implementation (`src/HansenRBC.jl`)

##### Core Model Types
- `RBCParams`: Model parameters (β, α, δ, B, h_bar) with validation
- `AROneProcess`: Discretized AR(1) process using Tauchen's method
- `RBCEndogenousVar`: Endogenous variables (policy functions, value functions)
- `HansenRBCModel`: Complete model combining parameters and process
- `VFIResult`: Solution results with convergence status and timing

##### Simulation and Counterfactual Types (NEW)
- `RBCSteadyState`: Steady state values (k*, h*, c*, y*, z̄)
- `ShockSpec`: Flexible shock specification (standard deviations, custom series)
- `RBCAggregateVars`: Aggregate time series (C, K, Y, H, Z)
- `ShockSeries`: Realized shock series with state indices and values
- `RBCSimulationResult`: Complete simulation results with aggregates and metadata

#### Tauchen Method (`src/tauchen.jl`)
- `tauchen()`: Function to discretize AR(1) processes into finite-state Markov chains
- Returns transition probability matrix and state vector
- Supports logarithmic transformation of states

## Common Development Commands

### Running Tests
```julia
using Pkg; Pkg.test("MacroProg")
```

### Running Individual Test Files
```julia
include("tests/test_tauchen.jl")
include("tests/stochasticGrowth.jl")
```

### Development Environment Setup
The project uses `Revise.jl` for development - include it when modifying code:
```julia
using Revise
using MacroProg
```

### Example Usage

#### Basic Model Solving
```julia
# Create model parameters
params = RBCParams(0.98, 0.36, 0.1, 1.0, 0.33)

# Create stochastic process
process = AROneProcess(0.9, 0.01, 0.0; N=7, m=3)

# Create capital grid
k_0 = collect(0.4:0.1:16)

# Initialize and solve model
model = HansenRBCModel(params, process, k_0)
solve_model!(model; tol=1e-6, max_iter=1000)
```

#### Simulation and Counterfactual Analysis (NEW)
```julia
# Compute steady state
steady_state = compute_steady_state(params, process)

# Baseline simulation
T = 200
baseline_result = simulate_model(model, T, burn_in=50, seed=1234)

# Impulse response function (+1σ technology shock)
irf_result = compute_irf(model, 1.0, horizon=40)

# Custom shock specification
custom_shocks = randn(T) .* process.σ .+ process.μ
custom_spec = ShockSpec(:custom, 0.0, custom_series=custom_shocks)
custom_result = simulate_model(model, T, spec=custom_spec, burn_in=0)

# Non-steady state initialization
k_init_high = steady_state.k_star * 1.5
nonss_result = simulate_model(model, T, k_init=k_init_high, burn_in=0)
```

## Key Dependencies

- **Optim.jl**: Used for numerical optimization in policy function computation
- **Interpolations.jl**: Linear interpolation for value functions
- **Roots.jl**: Root-finding for labor supply FOCs
- **Distributions.jl**: Statistical distributions for Tauchen method
- **QuantEcon.jl**: Reference implementation for testing Tauchen method
- **BenchmarkTools.jl**: Performance benchmarking
- **Plots.jl**: Visualization tools (used in tests)
- **NonlinearSolve.jl**: Steady state computation using Newton-Raphson method (NEW)

## Solving Algorithm

### Value Function Iteration
The main solver uses value function iteration with the following key functions:
- `value_function_iteration()`: Main VFI loop with convergence checking
- `compute_optimal_choice()`: Optimizes next period capital and labor supply
- `compute_labor_given_k_next()`: Solves labor FOC given next period capital
- Uses L-BFGS-B optimization with grid search fallback

### Steady State Computation (NEW)
- `compute_steady_state()`: Computes deterministic steady state using nonlinear equations
- Uses `NonlinearSolve.jl` with Newton-Raphson method
- Solves for k*, h*, c*, y*, z̄ simultaneously
- Provides robust initial guesses and validation

### Simulation and Counterfactual Analysis (NEW)
- `generate_shock_series()`: Creates realized shock paths from AR(1) process
- `simulate_model()`: Main simulation routine with flexible initialization
- `compute_irf()`: Impulse response function computation
- `compute_aggregates()`: Converts individual policies to aggregate time series
- Supports multiple shock specifications and non-steady state initialization

## Testing Structure

### Core Tests
- `tests/test_tauchen.jl`: Validates Tauchen discretization against QuantEcon.jl reference
- `tests/stochasticGrowth.jl`: Standalone stochastic growth model implementation for comparison

### Simulation and Counterfactual Tests (NEW)
- `tests/simulation_test.jl`: Comprehensive test suite for simulation extensions
  - Tests baseline simulation, impulse response functions
  - Custom shock specifications and non-steady state convergence
  - Includes visualizations and summary statistics
  - 12 different test scenarios covering all simulation functionality

## Current Development Stage

**Status**: ✅ SIMULATION SYSTEM COMPLETE
**Last Updated**: 2025-11-22

### Recent Implementation
The project now includes a complete simulation and counterfactual analysis system:

1. **Steady State Computation**: Deterministic steady state calculation using robust numerical methods
2. **Flexible Shock System**: Support for standard deviation-based counterfactuals and custom shock series
3. **Aggregate Variables**: Complete time series computation for C, K, Y, H, Z
4. **Simulation Pipeline**: Full simulation workflow with non-steady state initialization
5. **Impulse Response Functions**: Standard IRF computation with flexible horizons
6. **Comprehensive Testing**: 12-test validation suite with visualizations

### Architecture Notes
- Uses abstract type hierarchy for future extensibility to other model types
- `RBCSimulationResult` serves as temporary concrete implementation
- Designed for easy refactoring to accommodate additional model classes
- Maintains consistency with existing VFI solver architecture

### Next Steps (User-Driven)
- User validation through running `tests/simulation_test.jl`
- Potential refactoring for broader model compatibility
- Additional shock types or policy analysis tools as needed