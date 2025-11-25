using LinearAlgebra, Interpolations, NLsolve

# ================= COLEMAN POLICY ITERATION =================

"""
    coleman_operator!(new_policy_k::Matrix{Float64}, new_policy_h::Matrix{Float64},
                      policy_k::Matrix{Float64}, policy_h::Matrix{Float64},
                      model::HansenRBCModel)::Nothing

Apply Coleman operator to update policy functions using Euler equations.

The Coleman operator finds the optimal policy functions that satisfy:
1. Capital Euler equation: u'(c_t) = β * E[u'(c_{t+1}) * (α*z_{t+1}*k_{t+1}^(α-1)*(h_{t+1}-h̄)^(1-α) + 1-δ)]
2. Labor FOC: (1-α)*z_t*k_t^α*(h_t-h̄)^(-α) = B/(1-h_t) * u'(c_t)

# Arguments
- `new_policy_k`: Output matrix for updated capital policy function
- `new_policy_h`: Output matrix for updated labor policy function
- `policy_k`: Current capital policy function
- `policy_h`: Current labor policy function
- `model`: HansenRBCModel with parameters and process

"""
function coleman_operator!(new_policy_k::Matrix{Float64}, new_policy_h::Matrix{Float64},
                          policy_k::Matrix{Float64}, policy_h::Matrix{Float64},
                          model::HansenRBCModel)::Nothing

    params = model.params
    process = model.process
    k_grid = model.k_0
    n_grid = length(k_grid)
    n_states = process.N
    β, α, δ, B, h_bar = params.β, params.α, params.δ, params.B, params.h_bar
    p = process.p
    states = process.states

    # Create interpolation objects for current policies
    k_interps = [LinearInterpolation(k_grid, policy_k[:, i_z], extrapolation_bc=Line()) for i_z in 1:n_states]
    h_interps = [LinearInterpolation(k_grid, policy_h[:, i_z], extrapolation_bc=Line()) for i_z in 1:n_states]

    # Loop through all grid points and states
    for i_k in 1:n_grid
        k_t = k_grid[i_k]

        for i_z in 1:n_states
            z_t = states[i_z]

            # Solve for optimal (k_next, h_t) given current policies
            optimal_k_next, optimal_h = solve_coleman_optimal_choice(
                k_t, z_t, i_z, k_interps, h_interps, params, process, k_grid
            )

            new_policy_k[i_k, i_z] = optimal_k_next
            new_policy_h[i_k, i_z] = optimal_h
        end
    end
end

"""
    solve_coleman_optimal_choice(k_t::Float64, z_t::Float64, z_idx::Int,
                                 k_interps::Vector, h_interps::Vector,
                                 params::RBCParams, process::AROneProcess,
                                 k_grid::Vector{Float64})::Tuple{Float64, Float64}

Solve for optimal choices (k_next, h) using Coleman approach.

This solves the system of Euler equations directly rather than maximizing value functions.
"""
function solve_coleman_optimal_choice(k_t::Float64, z_t::Float64, z_idx::Int,
                                     k_interps::Vector, h_interps::Vector,
                                     params::RBCParams, process::AROneProcess,
                                     k_grid::Vector{Float64})::Tuple{Float64, Float64}

    β, α, δ, B, h_bar = params.β, params.α, params.δ, params.B, params.h_bar
    p = process.p
    n_states = process.N
    states = process.states

    function coleman_conditions!(F, x)
        k_next, h = x[1], x[2]

        # Resource constraint
        y_t = z_t * k_t^α * (h - h_bar)^(1 - α)
        c_t = y_t + (1 - δ) * k_t - k_next

        # Safety checks
        if c_t <= 1e-8 || h <= h_bar || h >= 1.0 || k_next <= k_grid[1] || k_next >= k_grid[end]
            F[1] = 1e8  # Large penalty
            F[2] = 1e8
            return F
        end

        # Current marginal utility
        u_c_t = 1.0 / c_t  # u'(c) = 1/c for log utility

        # Expected future marginal utility and capital return
        expected_u_c_future = 0.0
        expected_r_future = 0.0

        for i_z_prime in 1:n_states
            z_prime = states[i_z_prime]
            weight = p[z_idx, i_z_prime]

            # Interpolate optimal future choices
            k_next_next = k_interps[i_z_prime](k_next)
            h_prime = h_interps[i_z_prime](k_next)

            # Safety for interpolation
            if k_next_next <= k_grid[1] || k_next_next >= k_grid[end] || h_prime <= h_bar || h_prime >= 1.0
                k_next_next = clamp(k_next_next, k_grid[1], k_grid[end])
                # k_next_next = clamp(k_next_next, k_grid[1], y_next + (1 - δ) * k_next - 1e-6)
                h_prime = clamp(h_prime, h_bar + 1e-6, 1.0 - 1e-6)
            end

            # Future production and consumption
            y_next = z_prime * k_next^α * (h_prime - h_bar)^(1 - α)
            c_next = y_next + (1 - δ) * k_next - k_next_next

            if c_next <= 1e-8
                c_next = 1e-8  # Safety
            end

            u_c_next = 1.0 / c_next
            r_next = α * z_prime * (h_prime - h_bar)^(1 - α) * k_next^(α - 1) - δ

            expected_u_c_future += weight * u_c_next
            expected_r_future += weight * r_next
        end

        # Capital Euler equation: u'(c_t) = β * E[u'(c_{t+1}) * (r_{t+1} + 1 - δ)]
        capital_euler = u_c_t - β * expected_u_c_future * (expected_r_future + 1 - δ)

        # Labor FOC: (1-α)*y_t/(h-h̄) = B/(1-h) * u'(c_t)
        labor_foc = (1 - α) * y_t / (h - h_bar) - B / (1 - h) * u_c_t

        F[1] = capital_euler
        F[2] = labor_foc
        return F
    end

    # Initial guess - use current policies or heuristic
    h_guess = max(h_bar + 0.1, 0.3)
    y_current = z_t * k_t^α * (h_guess - h_bar)^(1 - α)
    k_guess = min(k_t * 0.9, y_current * 0.8)
    k_guess = max(k_grid[1], min(k_grid[end], k_guess))

    x0 = [k_guess, h_guess]

    try
        # Use NLsolve to solve the system of Euler equations
        result = nlsolve(coleman_conditions!, x0, method=:newton, ftol=1e-8, iterations=50)

        if result.f_converged
            k_opt, h_opt = result.zero[1], result.zero[2]

            # Ensure feasibility
            k_opt = clamp(k_opt, k_grid[1], k_grid[end])
            h_opt = clamp(h_opt, h_bar + 1e-6, 1.0 - 1e-6)

            # Final feasibility check
            y_opt = z_t * k_t^α * (h_opt - h_bar)^(1 - α)
            c_opt = y_opt + (1 - δ) * k_t - k_opt
            if c_opt <= 1e-6
                # Fallback to feasible solution
                k_opt = min(k_opt, y_opt + (1 - δ) * k_t - 1e-6)
            end

            return k_opt, h_opt
        else
            # Fallback to heuristic solution
            return fallback_coleman_solution(k_t, z_t, params, k_grid)
        end
    catch e
        # Fallback to heuristic solution
        return fallback_coleman_solution(k_t, z_t, params, k_grid)
    end
end

"""
    fallback_coleman_solution(k_t::Float64, z_t::Float64, params::RBCParams,
                             k_grid::Vector{Float64})::Tuple{Float64, Float64}

Fallback solution using simple heuristics when NLsolve fails.
"""
function fallback_coleman_solution(k_t::Float64, z_t::Float64, params::RBCParams,
                                   k_grid::Vector{Float64})::Tuple{Float64, Float64}
    α, δ, B, h_bar = params.α, params.δ, params.B, params.h_bar

    # Simple heuristic: use steady state ratios adjusted by technology
    h_star_guess = max(h_bar + 0.1, 0.3)
    y_guess = z_t * k_t^α * (h_star_guess - h_bar)^(1 - α)

    # Conservative saving rate
    saving_rate = 0.2
    k_next_guess = min(k_grid[end], max(k_grid[1], (1 - saving_rate) * ((1 - δ) * k_t + y_guess)))

    return k_next_guess, h_star_guess
end

"""
    coleman_policy_iteration(model::HansenRBCModel; tol::Float64=1e-6, max_iter::Int=100)::VFIResult

Solve the RBC model using Coleman policy iteration (time iteration).

This method directly solves the Euler equations rather than value function maximization,
which is often faster and more accurate for smooth models.

# Returns
- `VFIResult`: Solution result with policy functions
"""
function coleman_policy_iteration(model::HansenRBCModel; tol::Float64=1e-6, max_iter::Int=100)::VFIResult
    n_states = model.n_states
    n_grid = model.n_grid
    k_grid = model.k_0
    params = model.params

    # Initialize policy functions with simple heuristics
    policy_k = zeros(n_grid, n_states)
    policy_h = zeros(n_grid, n_states)

    # Initialize with steady state heuristics
    for i_k in 1:n_grid, i_z in 1:n_states
        k = k_grid[i_k]
        z = model.process.states[i_z]

        # Simple heuristic for capital policy (save a fraction)
        policy_k[i_k, i_z] = k * 0.95
        policy_h[i_k, i_z] = max(params.h_bar + 0.1, 0.3)
    end

    new_policy_k = similar(policy_k)
    new_policy_h = similar(policy_h)

    n_iter = 0
    diff = Inf
    time_used = @elapsed begin
        for iter in 1:max_iter
            n_iter += 1

            # Apply Coleman operator
            coleman_operator!(new_policy_k, new_policy_h, policy_k, policy_h, model)

            # Check convergence
            max_diff_k = maximum(abs.(new_policy_k .- policy_k))
            max_diff_h = maximum(abs.(new_policy_h .- policy_h))
            diff = max(max_diff_k, max_diff_h)

            # Update policies
            policy_k .= new_policy_k
            policy_h .= new_policy_h

            if diff < tol
                break
            end
        end
    end

    # Create dummy value function (not directly used in Coleman iteration)
    v_func = zeros(n_grid, n_states)

    # Compute simple value function estimate from policies for compatibility
    compute_value_from_policies!(v_func, policy_k, policy_h, model)

    endo_vars = RBCEndogenousVar(n_grid, n_states)
    update!(endo_vars, policy_k, policy_h, v_func)

    if diff < tol
        status = SolvingStatus(0, "Coleman policy iteration converged successfully in $n_iter iterations.")
    else
        status = SolvingStatus(1, "Coleman policy iteration did not converge within $max_iter iterations.")
    end

    return VFIResult(endo_vars; status=status, n_loops=n_iter, time_used=time_used)
end

"""
    compute_value_from_policies!(v_func::Matrix{Float64}, policy_k::Matrix{Float64},
                               policy_h::Matrix{Float64}, model::HansenRBCModel)::Nothing

Compute value function from given policy functions using backward induction.
"""
function compute_value_from_policies!(v_func::Matrix{Float64}, policy_k::Matrix{Float64},
                                     policy_h::Matrix{Float64}, model::HansenRBCModel)::Nothing

    params = model.params
    process = model.process
    k_grid = model.k_0
    n_grid, n_states = size(policy_k)
    β, α, δ, B, h_bar = params.β, params.α, params.δ, params.B, params.h_bar
    p = process.p

    # Simple value computation using fixed point iteration
    v_func_new = similar(v_func)

    for iter in 1:50  # Fixed number of iterations for value computation
        for i_k in 1:n_grid
            k_t = k_grid[i_k]

            for i_z in 1:n_states
                z_t = process.states[i_z]
                k_next = policy_k[i_k, i_z]
                h_t = policy_h[i_k, i_z]

                # Compute consumption and utility
                y_t = z_t * k_t^α * (h_t - h_bar)^(1 - α)
                c_t = y_t + (1 - δ) * k_t - k_next

                if c_t <= 1e-8 || h_t <= h_bar || h_t >= 1.0
                    u_t = -1e10  # Large penalty
                else
                    u_t = log(c_t) + B * log(1 - h_t)
                end

                # Interpolate next period value
                if k_next <= k_grid[1]
                    v_next = v_func[1, :]
                elseif k_next >= k_grid[end]
                    v_next = v_func[end, :]
                else
                    idx = searchsortedfirst(k_grid, k_next)
                    if idx >= 2 && idx <= n_grid
                        k_lower, k_upper = k_grid[idx-1], k_grid[idx]
                        v_lower, v_upper = v_func[idx-1, :], v_func[idx, :]
                        weight = (k_next - k_lower) / (k_upper - k_lower)
                        v_next = v_lower + weight * (v_upper - v_lower)
                    else
                        v_next = v_func[min(idx, n_grid), :]
                    end
                end

                # Expected future value
                ev_next = sum(p[i_z, :] .* v_next)
                v_func_new[i_k, i_z] = u_t + β * ev_next
            end
        end
        v_func .= v_func_new
    end
end