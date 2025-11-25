using LinearAlgebra, Optim, Interpolations, Roots, NLsolve, Random
using SparseArrays, Logging

# construct a type for solving status: 0-converged, 1-not converged, 2-error
struct SolvingStatus
    status::Int
    message::String
    function SolvingStatus(status::Int, message::String)
        new(status, message)
    end
end
SolvingStatus() = SolvingStatus(-1, "Not solved yet.")

"""
Params: Model (economic agents') parameters for Hansen's RBC model.

# Fields
- `β::Float64`: Discount factor.
- `α::Float64`: Capital share of income.
- `δ::Float64`: Depreciation rate.
- `B::Float64`: Parameter for simple log utility function
- `h_bar::Float64`: Minimum labor supply.
"""
mutable struct RBCParams <: AbstractParams
    β::Float64
    α::Float64
    δ::Float64
    B::Float64
    h_bar::Float64
    function RBCParams(β::Float64, α::Float64, δ::Float64, B::Float64, h_bar::Float64)
        @assert 0 < β < 1 "Discount factor β must be in (0, 1)."
        @assert 0 < α < 1 "Capital share α must be in (0, 1)."
        @assert 0 < δ < 1 "Depreciation rate δ must be in (0, 1)."
        @assert B > 0 "Parameter B must be positive."
        # @assert 0 < h_bar < 1 "Minimum labor h_bar must be in (0, 1)."
        new(β, α, δ, B, h_bar)
    end
end

"""
AROneProcess: Discretized AR(1) process 

z_{t+1} = ρ z_t + ε_t, ε_t ~ N(0, σ^2)

using Tauchen's method.

# Fields
- `ρ::Float64`: Autocorrelation coefficient.
- `σ::Float64`: Standard deviation of the error term.
- `μ::Float64`: Drift term of the process.
- `N::Int`: Number of discrete states.
- `m::Int`: Number of standard deviations to cover.
- `taking_log::Bool`: Whether to take the logarithm of the process states
"""
struct AROneProcess <: AbstractProcess
    ρ::Float64
    σ::Float64
    μ::Float64
    N::Integer
    m::Integer
    taking_log::Bool
    p::Matrix{Float64}
    states::Vector{Float64}
    function AROneProcess(ρ::Float64, σ::Float64, μ::Float64; N::Integer, m::Integer, taking_log::Bool=true)
        # add data validation
        @assert 0 <= ρ < 1 "Autocorrelation coefficient ρ must be in [0, 1)."
        @assert σ > 0 "Standard deviation σ must be positive."
        @assert N > 1 "Number of discrete states N must be greater than 1."
        @assert m > 0 "Number of standard deviations m must be positive."
        @assert μ >= 0 || taking_log "Mean μ must not be negative when taking_log is false."
        p, states = tauchen(ρ, σ, μ; N, m, taking_log)
        new(ρ, σ, μ, N, m, taking_log, p, states)
    end
end

mutable struct RBCEndogenousVar <: AbstractVariables
    n_grid::Integer
    n_state::Integer
    policy_k::Matrix{Float64}
    policy_h::Matrix{Float64}
    policy_c::Matrix{Float64}
    v_func::Matrix{Float64}
    function RBCEndogenousVar(n_grid::Integer, n_state::Integer)
        policy_k = zeros(n_grid, n_state)
        policy_h = zeros(n_grid, n_state)
        policy_c = zeros(n_grid, n_state)
        v_func = zeros(n_grid, n_state)
        new(n_grid, n_state, policy_k, policy_h, policy_c, v_func)
    end
end

struct VFIResult <: AbstractSolvingResult
    endo_vars::RBCEndogenousVar
    status::SolvingStatus
    n_loops::Integer
    time_used::Float64
    function VFIResult(endo_vars::RBCEndogenousVar; status::SolvingStatus, n_loops::Integer, time_used::Float64)
        new(endo_vars, status, n_loops, time_used)
    end
end

mutable struct HansenRBCModel <: AbstractModel
    params::RBCParams
    process::AROneProcess
    k_0::Vector{Float64}
    n_states::Integer
    n_grid::Integer
    endogenous_vars::RBCEndogenousVar
    solving_result::AbstractSolvingResult
    function HansenRBCModel(params::RBCParams, process::AROneProcess, k_0::Vector{Float64})
        n_states = process.N
        n_grid = length(k_0)
        endogenous_vars = RBCEndogenousVar(n_grid, n_states)
        new(params, process, k_0, n_states, n_grid, endogenous_vars)
    end
end

function update!(vars::RBCEndogenousVar, policy_k::Matrix{Float64}, policy_c::Matrix{Float64}, policy_h::Matrix{Float64}, v_func::Matrix{Float64})
    vars.policy_k .= policy_k
    vars.policy_c .= policy_c
    vars.policy_h .= policy_h
    vars.v_func .= v_func
    return nothing
end

function utility(params::RBCParams, c::Float64, h::Float64)::Float64
    B = params.B
    @assert c > 0 "Consumption c must be positive."
    @assert 0 <= h < 1 "Labor h must be in [0, 1)."
    return log(c) + B * log(1 - h)
end

function production(params::RBCParams, k::Float64, z::Float64, h::Float64)::Float64
    α = params.α
    h_bar = params.h_bar
    @assert k >= 0 "Capital k must be non-negative."
    @assert z > 0 "Technology shock z must be positive."
    @assert h_bar <= h <= 1 "Labor h must be in [h_bar, 1]."
    return z * k^α * (h - h_bar)^(1 - α)
end

function grad_utility(params::RBCParams, k::Float64, k_next::Float64, h::Float64, z::Float64)::Tuple{Float64,Float64}
    B = params.B
    δ = params.δ
    α = params.α
    h_bar = params.h_bar
    consumption = production(params, k, z, h) + (1 - δ) * k - k_next
    du_dk_next = -1 / consumption
    du_dh = ((1 - α) * production(params, k, z, h) / (h - h_bar)) / consumption - B / (1 - h)
    return du_dk_next, du_dh
end

function foc_labor(params::RBCParams, k::Float64, k_next::Float64, h::Float64, z::Float64)::Float64
    B = params.B
    δ = params.δ
    α = params.α
    h_bar = params.h_bar
    result = (1 - α) * production(params, k, z, h) * (1 - h) / (h - h_bar) - B * (production(params, k, z, h) + (1 - δ) * k - k_next)
    return result
end

function grad_foc_labor(params::RBCParams, k::Float64, h::Float64, z::Float64)::Float64
    B = params.B
    δ = params.δ
    α = params.α
    h_bar = params.h_bar
    term_1 = -(1 - α) * production(params, k, z, h) / (h - h_bar)
    term_2 = -α * (1 - α) * production(params, k, z, h) * (1 - h) / (h - h_bar)^2
    term_3 = -B * (1 - α) * production(params, k, z, h) / (h - h_bar)
    result = term_1 + term_2 + term_3
    return result
end

function compute_labor_given_k_next(params::RBCParams, k::Float64, k_next::Float64, z::Float64)::Float64

    f(x) = foc_labor(params, k, k_next, x, z)
    fprime(x) = grad_foc_labor(params, k, x, z)
    x0 = (params.h_bar + 1.0) / 2
    h_star = find_zero((f, fprime), x0, Roots.Newton(); tol=1e-12, maxevals=200)
    return h_star

end

function vfi_optimal_choice(params::RBCParams, k::Float64, z::Float64, process::AROneProcess, k_0::Vector{Float64}, v_func::Matrix{Float64})::Tuple{Float64,Float64}
    β = params.β
    h_bar = params.h_bar
    δ = params.δ
    states = process.states
    n_states = process.N
    n_grids = length(k_0)
    p = process.p
    which_state = findfirst(x -> x == z, states)
    p_vec = p[which_state, :]

    function f(x::Vector{Float64})::Float64
        k_next = x[1]
        h = x[2]
        if h <= h_bar || h >= 1
            return Inf
        end
        resource = production(params, k, z, h) + (1 - δ) * k
        c = resource - k_next
        if c <= 1e-6 || k_next < k_0[1] || k_next > k_0[end]
            return Inf
        end
        u = utility(params, c, h)
        idx = searchsortedfirst(k_0, k_next)
        if idx >= 2
            v_func_interval = v_func[(idx-1):idx, :]
            k_0_interval = k_0[(idx-1):idx]
            v_next = (k_next - k_0_interval[1]) .* (v_func_interval[2, :] - v_func_interval[1, :]) ./ (k_0_interval[2] - k_0_interval[1]) .+ v_func_interval[1, :]
        else
            v_next = v_func[1, :]
        end
        ev = sum(p_vec .* v_next)
        return -(u + params.β * ev)
    end

    function g!(G::Vector{Float64}, x::Vector{Float64})
        k_next = x[1]
        h = x[2]

        if h <= h_bar || h >= 1
            G[1] = 0.0
            G[2] = 0.0
            return
        end

        du_dk_next, du_dh = grad_utility(params, k, k_next, h, z)

        idx = searchsortedfirst(k_0, k_next)
        if idx >= 2
            v_func_interval = v_func[(idx-1):idx, :]
            k_0_interval = k_0[(idx-1):idx]
            dv_next_dk_next_i = (1 / (k_0_interval[2] - k_0_interval[1])) .* (v_func_interval[2, :] - v_func_interval[1, :])
        else
            dv_next_dk_next_i = zeros(n_states)
        end

        dv_next_dk_next = sum(p_vec .* dv_next_dk_next_i)
        G[1] = -du_dk_next - β * dv_next_dk_next
        G[2] = -du_dh
    end

    # Calculate maximum feasible k_next based on resource constraint
    # For initial guess, use moderate labor supply to estimate max feasible capital
    h_guess = (h_bar + 1.0) / 2
    resource_max = production(params, k, z, h_guess) + (1 - δ) * k
    k_next_max = min(k_0[end], resource_max - 1e-6)  # Ensure positive consumption

    lower = [k_0[1], h_bar]
    upper = [k_next_max, 1.0]
    x0 = [min((k_0[1] + k_next_max) / 2, k_next_max * 0.8), (h_bar + 1.0) / 2]
    opt_alg = Fminbox(LBFGS())
    result = optimize(f, g!, lower, upper, x0, opt_alg)

    if Optim.converged(result)
        res = Optim.minimizer(result)
        best_k_next = res[1]
        best_h = res[2]
        return best_k_next, best_h
    else
        # fallback to loop k_next and compute h_star
        best_value = -Inf
        for i in 1:length(k_0)
            k_next_candidate = k_0[i]
            h_star = compute_labor_given_k_next(params, k, k_next_candidate, z)
            resource = production(params, k, z, h_star) + (1 - δ) * k
            c = resource - k_next_candidate
            if h_star < h_bar || c <= 1e-6 || h_star >= 1
                continue
            end
            u = utility(params, c, h_star)
            idx = searchsortedfirst(k_0, k_next_candidate)
            if idx >= 2
                v_func_interval = v_func[(idx-1):idx, :]
                k_0_interval = k_0[(idx-1):idx]
                v_next = (k_next_candidate - k_0_interval[1]) .* (v_func_interval[2, :] - v_func_interval[1, :]) ./ (k_0_interval[2] - k_0_interval[1]) .+ v_func_interval[1, :]
            else
                v_next = v_func[1, :]
            end
            ev = sum(p_vec .* v_next)
            total_value = u + params.β * ev
            if total_value > best_value
                best_value = total_value
                best_k_next = k_next_candidate
                best_h = h_star
            end
        end
        return best_k_next, best_h
    end

end

function pfi_optimal_choice(params::RBCParams, k::Float64, z_loc::Int, process::AROneProcess, k_0::Vector{Float64}, policy_c::Matrix{Float64}, policy_h::Matrix{Float64})::Tuple{Float64,Float64}

    β = params.β
    h_bar = params.h_bar
    δ = params.δ
    α = params.α
    B = params.B

    states = process.states
    n_states = process.N
    p_vec = process.p[z_loc, :]
    z = states[z_loc]

    function policy_intern(k_next::Float64, z_idx::Integer, policy_grid::Matrix{Float64})::Float64
        k_loc = searchsortedfirst(k_0, k_next)
        bound = [1, length(k_0)]
        if k_loc == bound[1]
            policy_next = policy_grid[bound[1], z_idx]
        elseif k_loc >= bound[2]
            policy_next = policy_grid[bound[2], z_idx]
        else
            policy_next = (k_next - k_0[k_loc-1]) * (policy_grid[k_loc, z_idx] - policy_grid[k_loc-1, z_idx]) / (k_0[k_loc] - k_0[k_loc-1]) + policy_grid[k_loc-1, z_idx]
        end
        return policy_next
    end

    function eq_1(c::Float64, h::Float64)::Float64
        production_term = k^α * (h - h_bar)^(1 - α) * z
        k_next = production_term * states .+ (1 - δ) * k .- c
        c_next = [policy_intern(k_next[j], j, policy_c) for j in 1:n_states]
        h_next = [policy_intern(k_next[j], j, policy_h) for j in 1:n_states]
        eq1 = 1.0 / c - β * sum(((α .* states .* k_next .^ (α - 1) .* (h_next .- h_bar) .^ (1 - α) .+ (1 - δ)) ./ c_next) .* p_vec)
        return eq1
    end

    function eq_2(c::Float64, h::Float64)::Float64
        eq2 = (1.0 - α) * z * k^α * (h - h_bar)^(-α) - (B / (1.0 - h)) * c
        return eq2
    end

    function f(x::Vector{Float64})::Float64
        c = x[1]
        h = x[2]
        if c <= 1e-6 || h <= h_bar || h >= 1
            return Inf
        end
        res = (eq_1(c, h)^2 + eq_2(c, h)^2) / 2
        return res
    end

    # Initial guess
    h0 = 0.5 * (h_bar + 1.0)
    c0 = minimum(k^α * (h0 - h_bar)^(1 - α) .* states .+ (1 - δ) * k) * 0.8
    c_max = minimum(k^α * (1 - h_bar)^(1 - α) .* states .+ (1 - δ) * k) - 1e-6
    x0 = [c0, h0]
    lower = [1e-6, h_bar]
    upper = [c_max, 1.0]
    result = optimize(f, lower, upper, x0, Fminbox(LBFGS()))

    res = Optim.minimizer(result)
    best_c = res[1]
    best_h = res[2]
    return best_c, best_h
end


"""
    compute_value_linear_solve_from_policies(k_grid, policy_c, policy_h, process, params; tol_check=1e-12)

Given converged policy_c and policy_h (size nk x nz), build the sparse operator P consistent
with your `pfi_optimal_choice` conventions and solve (I - β P) V = u for V.

Returns:
    V :: Array{Float64}(nk, nz)
    kprime :: Array{Float64}(nk, nz)   # implied next-capital at each (ik, iz) for bookkeeping
"""
function compute_value_linear_solve_from_policies(
    k_grid::Vector{Float64},
    policy_c::Array{Float64,2},
    policy_h::Array{Float64,2},
    process,
    params;
    tol_check::Float64=1e-12
)
    nk = length(k_grid)
    nz = length(process.states)
    N = nk * nz

    # flatten index: z-major ordering like your code (for iz=1:nz, ik=1:nk)
    flat_idx(ik, iz) = (iz - 1) * nk + ik

    # shortcuts to params used in pfi_optimal_choice
    β = params.β
    h_bar = params.h_bar
    δ = params.δ
    α = params.α
    B = params.B

    # 1) build immediate utility vector uvec of length N
    uvec = similar(zeros(Float64, N))
    for iz in 1:nz
        for ik in 1:nk
            idx = flat_idx(ik, iz)
            c = policy_c[ik, iz]
            h = policy_h[ik, iz]

            # feasibility guard
            if !(c > 1e-12) || !(h > h_bar) || !(h < 1.0)
                uvec[idx] = -1e12
            else
                # utility: log(c) + B * log(1 - h)
                uvec[idx] = log(c) + B * log(1.0 - h)
            end
        end
    end

    # 2) compute implied k' at each (ik, iz) following your pfi code:
    #    production_term = k^α * (h - h_bar)^(1 - α) * z_current
    #    then k'_j = production_term * process.states[j] + (1-δ)*k - c
    kprime = zeros(Float64, nk, nz)
    for iz in 1:nz
        zcurr = process.states[iz]
        for ik in 1:nk
            k = k_grid[ik]
            c = policy_c[ik, iz]
            h = policy_h[ik, iz]

            production_term = (k^α) * (h - h_bar)^(1.0 - α) * zcurr
            # produce k' for each possible next-state -> we only store the one for current iz here,
            # but later when assembling P we will compute weights over next-state indices.
            # store the *expected* or representative k' for reference (we still need per-izp values when
            # assembling P so we recompute there).
            kprime[ik, iz] = (1.0 - δ) * k + production_term * process.states[1] - c  # placeholder; overwritten below if needed
            # (We will compute per-izp kp when assembling P below.)
        end
    end

    # 3) assemble sparse operator P with same interpolation/boundary logic as your policy_intern
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for iz in 1:nz
        zcurr = process.states[iz]
        for ik in 1:nk
            row = flat_idx(ik, iz)
            k = k_grid[ik]
            c = policy_c[ik, iz]
            h = policy_h[ik, iz]
            production_term = (k^α) * (h - h_bar)^(1.0 - α) * zcurr

            # for each possible next-state izp compute kp and interpolation weights
            for izp in 1:nz
                kp = production_term * process.states[izp] + (1.0 - δ) * k - c
                # guard against extremely small/negative kp
                if kp < 1e-12
                    kp = 1e-12
                end

                # locate kp on k_grid using same style as your policy_intern
                j = searchsortedfirst(k_grid, kp)  # first index >= kp
                if j == 1
                    jL, jR = 1, min(2, nk)
                elseif j > nk
                    jL, jR = max(1, nk-1), nk
                else
                    jR = j
                    jL = j - 1
                end

                kL, kR = k_grid[jL], k_grid[jR]
                if kR == kL
                    wL, wR = 1.0, 0.0
                else
                    wR = (kp - kL) / (kR - kL)
                    wL = 1.0 - wR
                end

                pzz = process.p[iz, izp]   # transition prob from iz -> izp
                colL = flat_idx(jL, izp)
                colR = flat_idx(jR, izp)

                push!(rows, row); push!(cols, colL); push!(vals, pzz * wL)
                push!(rows, row); push!(cols, colR); push!(vals, pzz * wR)
            end
        end
    end

    P = sparse(rows, cols, vals, N, N)

    # 4) solve (I - β P) V = uvec
    A = I - β * P     # supports sparse P
    Vflat = A \ uvec

    # 5) diagnostic residual
    resid = uvec + β * (P * Vflat) - Vflat
    maxresid = maximum(abs.(resid))
    if maxresid > tol_check
        @warn "High residual in policy evaluation" maxresid=maxresid
    end

    V = reshape(Vflat, nk, nz)

    # recompute kprime per (ik, iz) for returning (useful for policy_k)
    for iz in 1:nz
        zcurr = process.states[iz]
        for ik in 1:nk
            k = k_grid[ik]
            c = policy_c[ik, iz]
            h = policy_h[ik, iz]
            production_term = (k^α) * (h - h_bar)^(1.0 - α) * zcurr
            kprime[ik, iz] = maximum([production_term * process.states[1] + (1.0 - δ) * k - c, 1e-12]) # just a stored representative; user may want per-izp values
        end
    end

    return V, kprime
end


function value_function_iteration(model::HansenRBCModel; tol::Float64=1e-6, max_iter::Int=1000)::VFIResult
    n_states = model.n_states
    n_grids = model.n_grid
    k_0 = model.k_0
    process = model.process
    params = model.params

    v_func = zeros(n_grids, n_states)
    policy_k = zeros(n_grids, n_states)
    policy_c = zeros(n_grids, n_states)
    policy_h = zeros(n_grids, n_states)
    n_iter = 0
    diff = Inf
    endo_vars = RBCEndogenousVar(n_grids, n_states)
    try
        time_used = @elapsed begin
            for iter in 1:max_iter
                n_iter += 1
                v_func_new = zeros(n_grids, n_states)
                for i_k in 1:n_grids
                    k = k_0[i_k]
                    for i_z in 1:n_states
                        z = process.states[i_z]
                        # println("yes")
                        best_k_next, best_h = vfi_optimal_choice(params, k, z, process, k_0, v_func)
                        # println("hello")
                        resource = production(params, k, z, best_h) + (1 - params.δ) * k
                        c = resource - best_k_next
                        u = utility(params, c, best_h)
                        idx = searchsortedfirst(k_0, best_k_next)
                        if idx >= 2
                            v_func_interval = v_func[(idx-1):idx, :]
                            k_0_interval = k_0[(idx-1):idx]
                            v_next = (best_k_next - k_0_interval[1]) .* (v_func_interval[2, :] - v_func_interval[1, :]) ./ (k_0_interval[2] - k_0_interval[1]) .+ v_func_interval[1, :]
                        else
                            v_next = v_func[1, :]
                        end
                        ev = sum(process.p[i_z, :] .* v_next)
                        v_func_new[i_k, i_z] = u + params.β * ev
                        policy_k[i_k, i_z] = best_k_next
                        policy_h[i_k, i_z] = best_h
                    end
                end
                diff = maximum(abs.(v_func_new .- v_func))
                v_func .= v_func_new
                if diff < tol
                    break
                end
            end
        end
        k_mat = repeat(k_0, 1, n_states)
        z_mat = repeat(process.states', n_grids, 1)
        policy_c .= k_mat .^ params.α .* (policy_h .- params.h_bar) .^ (1 - params.α) .+ (1 - params.δ) .* z_mat .- policy_k .+ (1 - params.δ) .* k_mat
        update!(endo_vars, policy_k, policy_c, policy_h, v_func)
        if diff < tol
            return VFIResult(
                endo_vars;
                status=SolvingStatus(0, "Converged successfully."),
                n_loops=n_iter,
                time_used=time_used
            )
        else
            return VFIResult(
                endo_vars;
                status=SolvingStatus(1, "Did not converge within max_iter."),
                n_loops=n_iter,
                time_used=time_used
            )
        end
    catch e
        return VFIResult(
            endo_vars;
            status=SolvingStatus(2, "Error during VFI: $(e.msg)"),
            n_loops=n_iter,
            time_used=0.0
        )
    end
end

function time_iteration(model::HansenRBCModel; tol::Float64=1e-6, max_iter::Int=1000)::VFIResult
    n_states = model.n_states
    n_grids = model.n_grid
    k_0 = model.k_0
    process = model.process
    params = model.params

    v_func = zeros(n_grids, n_states)
    policy_k = zeros(n_grids, n_states)
    policy_c = repeat(k_0, 1, n_states)
    # policy_c multiples a vector from 0.25 to 0.75
    policy_c .= policy_c .* reshape(range(0.25, 0.75, length=n_grids), n_grids, 1)
    policy_c .= policy_c .* reshape(range(0.7, 1.4, length=n_states), 1, n_states)
    policy_h = ones(n_grids, n_states)
    policy_h = policy_h .* reshape(range(0.8, params.h_bar+0.1, length=n_grids), n_grids, 1)
    n_iter = 0
    diff = Inf
    endo_vars = RBCEndogenousVar(n_grids, n_states)
    try
        time_used = @elapsed begin
            for iter in 1:max_iter
                n_iter += 1
                policy_c_new = zeros(n_grids, n_states)
                policy_h_new = zeros(n_grids, n_states)
                for i_k in 1:n_grids
                    k = k_0[i_k]
                    for i_z in 1:n_states
                        # z = process.states[i_z]
                        best_c, best_h = pfi_optimal_choice(params, k, i_z, process, k_0, policy_c, policy_h)
                        policy_c_new[i_k, i_z] = best_c
                        policy_h_new[i_k, i_z] = best_h
                    end
                end
                diff = maximum(abs.(policy_c_new .- policy_c)) # add policy_h difference 
                diff = max(diff, maximum(abs.(policy_h_new .- policy_h)))
                policy_c .= policy_c_new
                policy_h .= policy_h_new

                if diff < tol
                    break
                end
            end
        end
        if diff < tol
            k_mat = repeat(k_0, 1, n_states)
            z_mat = repeat(process.states', n_grids, 1)
            policy_k .= k_mat .^ params.α .* (policy_h .- params.h_bar) .^ (1 - params.α) .+ (1 - params.δ) .* z_mat .- policy_c .+ (1 - params.δ) .* k_mat
            V_matrix, kprime = compute_value_linear_solve_from_policies(k_0, policy_c, policy_h, process, params)
            update!(endo_vars, kprime, policy_c, policy_h, V_matrix)
            return VFIResult(
                endo_vars;
                status=SolvingStatus(0, "Converged successfully."),
                n_loops=n_iter,
                time_used=time_used
            )
        else
            return VFIResult(
                endo_vars;
                status=SolvingStatus(1, "Did not converge within max_iter."),
                n_loops=n_iter,
                time_used=time_used
            )
        end

    catch e
        return VFIResult(
            endo_vars;
            status=SolvingStatus(2, "Error during time iteration: $(e.msg)"),
            n_loops=n_iter,
            time_used=0.0
        )
    end

end

function solve_model!(model::HansenRBCModel; method::String="VFI", tol::Float64=1e-6, max_iter::Int=1000)
    if method == "VFI"
        result = value_function_iteration(model; tol=tol, max_iter=max_iter)
    elseif method == "TI"
        result = time_iteration(model; tol=tol, max_iter=max_iter)
    end
    model.solving_result = result
    model.endogenous_vars = result.endo_vars
    return nothing
end

# ================= SIMULATION EXTENSIONS =================

"""
RBCSteadyState: Steady state equilibrium for Hansen RBC model

# Fields
- `k_star::Float64`: Steady state capital stock
- `h_star::Float64`: Steady state labor supply
- `c_star::Float64`: Steady state consumption
- `y_star::Float64`: Steady state output
- `z_bar::Float64`: Mean technology level
- `r_star::Float64`: Steady state real interest rate
- `w_star::Float64`: Steady state real wage
"""
struct RBCSteadyState <: AbstractSteadyState
    k_star::Float64
    h_star::Float64
    c_star::Float64
    y_star::Float64
    z_bar::Float64
    r_star::Float64
    w_star::Float64
end

"""
ShockSpec: Specification for generating shock series

# Fields
- `shock_type::Symbol`: Type of shock (:technology, :custom, :baseline_deviation)
- `shock_size::Float64`: Size in standard deviations (for technology shocks)
- `timing::Int`: Period when shock hits (1 for immediate, >1 for delayed)
- `duration::Int`: Duration of shock (1 for one-time, >1 for persistent)
- `custom_series::Union{Vector{Float64}, Nothing}`: Custom shock series if provided
- `baseline_path::Union{Vector{Float64}, Nothing}`: Baseline path for deviations
"""
struct ShockSpec
    shock_type::Symbol  # :technology, :custom, :baseline_deviation
    shock_size::Float64
    timing::Int
    duration::Int
    custom_series::Union{Vector{Float64},Nothing}
    baseline_path::Union{Vector{Float64},Nothing}

    function ShockSpec(shock_type::Symbol, shock_size::Float64=0.0;
        timing::Int=1, duration::Int=1,
        custom_series::Union{Vector{Float64},Nothing}=nothing,
        baseline_path::Union{Vector{Float64},Nothing}=nothing)
        @assert shock_type in [:technology, :custom, :baseline_deviation] "Invalid shock type"
        @assert timing >= 1 "Timing must be >= 1"
        @assert duration >= 1 "Duration must be >= 1"
        new(shock_type, shock_size, timing, duration, custom_series, baseline_path)
    end
end

"""
RBCAggregateVars: Aggregate variables for simulation output

# Fields
- `C::Vector{Float64}`: Aggregate consumption series
- `K::Vector{Float64}`: Aggregate capital series
- `H::Vector{Float64}`: Aggregate labor hours series
- `Y::Vector{Float64}`: Aggregate output series
- `Z::Vector{Float64}`: Technology shock series
- `R::Vector{Float64}`: Real interest rate series
- `W::Vector{Float64}`: Real wage series
- `T::Int`: Simulation length
"""
struct RBCAggregateVars <: AbstractAggregates
    C::Vector{Float64}
    K::Vector{Float64}
    H::Vector{Float64}
    Y::Vector{Float64}
    Z::Vector{Float64}
    R::Vector{Float64}
    W::Vector{Float64}
    T::Int
end

"""
ShockSeries: Realized shock series for simulation

# Fields
- `z_series::Vector{Float64}`: Technology shock series
- `state_indices::Vector{Int}`: Discrete state indices used
- `T::Int`: Length of shock series
- `seed::Int`: Random seed for reproducibility
- `spec::ShockSpec`: Specification used to generate this series
"""
struct ShockSeries <: AbstractShockSeries
    z_series::Vector{Float64}
    state_indices::Vector{Int}
    T::Int
    seed::Int
    spec::ShockSpec
end

"""
RBCSimulationResult: Results from RBC model simulation

# Fields
- `aggregates::RBCAggregateVars`: Aggregate variables
- `shocks::ShockSeries`: Shock series used
- `status::String`: Simulation status
- `converged::Bool`: Whether simulation converged
- `info::Dict{String, Any}`: Additional simulation information
"""
struct RBCSimulationResult <: AbstractSimulationResult
    aggregates::RBCAggregateVars
    shocks::ShockSeries
    status::String
    converged::Bool
    info::Dict{String,Any}
end

# ================= CORE SIMULATION FUNCTIONS =================

"""
    compute_steady_state(params::RBCParams, process::AROneProcess)::RBCSteadyState

Compute the steady state equilibrium of the Hansen RBC model analytically.

For the model with log utility and Cobb-Douglas production:
- u(c,h) = log(c) + B*log(1-h)
- y = z*k^α*(h-h̄)^(1-α)

# Returns
- `RBCSteadyState`: Steady state values for all variables
"""
function compute_steady_state(params::RBCParams, process::AROneProcess)::RBCSteadyState
    β, α, δ, B, h_bar = params.β, params.α, params.δ, params.B, params.h_bar

    # Mean technology level
    if process.taking_log
        z_bar = exp(process.μ / (1 - process.ρ))
    else
        z_bar = process.μ / (1 - process.ρ)
    end

    # Steady state from Euler equation and FOCs
    # Using numerical solving for robustness
    function steady_state_conditions(vars)
        k, h = vars[1], vars[2]

        # Euler equation: 1 = β * [α*z*(h-h̄)^(1-α)*k^(α-1) + 1 - δ]
        euler = 1 - β * (α * z_bar * (h - h_bar)^(1 - α) * k^(α - 1) + 1 - δ)

        # Labor FOC: (1-α)*z*k^α*(h-h̄)^(-α) = B/(1-h) * (z*k^α*(h-h̄)^(1-α) + (1-δ)*k - k)
        y = z_bar * k^α * (h - h_bar)^(1 - α)
        c = y - δ * k
        labor_foc = (1 - α) * y / (h - h_bar) - B * c / (1 - h)

        return [euler, labor_foc]
    end

    # Initial guesses
    k_guess = ((α * z_bar * β) / (1 / β - 1 + δ))^(1 / (1 - α))
    h_guess = max(h_bar + 0.1, 0.3)

    # Solve for steady state
    prob = NonlinearProblem(steady_state_conditions, [k_guess, h_guess])
    sol = solve(prob, NewtonRaphson(), abstol=1e-10, maxiters=100)
    k_star, h_star = sol.u[1], sol.u[2]

    # Compute other steady state variables
    y_star = z_bar * k_star^α * (h_star - h_bar)^(1 - α)
    c_star = y_star - δ * k_star
    r_star = α * z_bar * (h_star - h_bar)^(1 - α) * k_star^(α - 1) - δ
    w_star = (1 - α) * z_bar * k_star^α * (h_star - h_bar)^(-α)

    return RBCSteadyState(k_star, h_star, c_star, y_star, z_bar, r_star, w_star)
end

"""
    generate_shock_series(process::AROneProcess, T::Int, spec::ShockSpec; seed::Int=1234)::ShockSeries

Generate a shock series based on the specification.

# Arguments
- `process`: The AR(1) process for technology shocks
- `T`: Length of series to generate
- `spec`: Shock specification
- `seed`: Random seed for reproducibility

# Returns
- `ShockSeries`: Generated shock series with state indices
"""
function generate_shock_series(process::AROneProcess, T::Int, spec::ShockSpec; seed::Int=1234)::ShockSeries
    Random.seed!(seed)
    z_series = zeros(T)
    state_indices = zeros(Int, T)

    if spec.shock_type == :technology
        # Generate baseline stochastic process
        stationary_dist = compute_stationary_distribution(process.p)
        current_state = findall(cumsum(stationary_dist) .>= rand())[1]
        z_series[1] = process.states[current_state]
        state_indices[1] = current_state

        # Generate Markov chain with counterfactual shock
        for t in 2:T
            # Check if we're in shock period
            if t >= spec.timing && t < spec.timing + spec.duration
                # Apply counterfactual shock (modify transition probabilities)
                shock_effect = exp(spec.shock_size * process.σ)
                shocked_states = process.states * shock_effect

                # Find closest state to shocked value
                target_z = shocked_states[findfirst(process.states .== z_series[t-1])] * process.ρ +
                           process.μ + spec.shock_size * process.σ

                # Find nearest discretized state
                distances = abs.(process.states .- target_z)
                current_state = findmin(distances)[2]
                z_series[t] = process.states[current_state]
            else
                # Normal Markov transition
                current_state = findall(cumsum(process.p[current_state, :]) .>= rand())[1]
                z_series[t] = process.states[current_state]
            end
            state_indices[t] = current_state
        end

    elseif spec.shock_type == :custom
        # Use custom shock series
        @assert length(spec.custom_series) == T "Custom series length must match T"
        z_series .= spec.custom_series
        for t in 1:T
            # Find nearest discretized state
            distances = abs.(process.states .- z_series[t])
            state_indices[t] = findmin(distances)[2]
        end

    elseif spec.shock_type == :baseline_deviation
        # Generate baseline then add deviation
        baseline_spec = ShockSpec(:technology, 0.0)
        baseline_series = generate_shock_series(process, T, baseline_spec, seed=seed)

        if !isnothing(spec.baseline_path)
            z_series = spec.baseline_path .+ spec.shock_size * process.σ
        else
            z_series = baseline_series.z_series
            for t in spec.timing:min(spec.timing + spec.duration - 1, T)
                z_series[t] *= (1 + spec.shock_size * process.σ)
            end
        end

        # Update state indices
        for t in 1:T
            distances = abs.(process.states .- z_series[t])
            state_indices[t] = findmin(distances)[2]
        end
    end

    return ShockSeries(z_series, state_indices, T, seed, spec)
end

"""
    compute_stationary_distribution(P::Matrix{Float64})::Vector{Float64}

Compute the stationary distribution of a Markov chain.

# Arguments
- `P`: Transition probability matrix

# Returns
- `π`: Stationary distribution vector
"""
function compute_stationary_distribution(P::Matrix{Float64})::Vector{Float64}
    n = size(P, 1)
    # Solve (P' - I)π = 0 with constraint sum(π) = 1
    A = [P' - I; ones(1, n)]
    b = [zeros(n); 1]
    π = A \ b
    return π
end

"""
    interpolate_policy(policy_matrix::Matrix{Float64}, k_grid::Vector{Float64}, k::Float64)::Float64

Interpolate policy function at given capital level.

# Arguments
- `policy_matrix`: Policy function values for each (k, state)
- `k_grid`: Capital grid points
- `k`: Capital level to interpolate at

# Returns
- Interpolated policy value
"""
function interpolate_policy(policy_matrix::Vector{Float64}, k_grid::Vector{Float64}, k::Float64)::Float64
    # Add safety check for extreme values
    if k <= k_grid[1]
        return policy_matrix[1]
    elseif k >= k_grid[end]
        return policy_matrix[end]
    else
        # Linear interpolation
        idx = searchsortedfirst(k_grid, k)
        if idx == 1
            return policy_matrix[1]
        elseif idx > length(k_grid)
            return policy_matrix[end]
        else
            # Linear interpolation between idx-1 and idx
            k_lower, k_upper = k_grid[idx-1], k_grid[idx]
            pol_lower, pol_upper = policy_matrix[idx-1], policy_matrix[idx]
            weight = (k - k_lower) / (k_upper - k_lower)

            # Add safety check for interpolated values
            interpolated_value = pol_lower + weight * (pol_upper - pol_lower)

            # For capital policy, ensure it's within reasonable bounds
            if any(isnan, [interpolated_value]) || !isfinite(interpolated_value)
                @warn "Invalid interpolated policy value, using boundary fallback"
                return k >= (k_grid[1] + k_grid[end]) / 2 ? policy_matrix[end] : policy_matrix[1]
            end

            return interpolated_value
        end
    end
end

"""
    compute_aggregates(model::HansenRBCModel, k_path::Vector{Float64},
                      z_path::Vector{Float64}, state_indices::Vector{Int})::RBCAggregateVars

Compute aggregate variables from simulated capital and technology paths.

# Arguments
- `model`: Solved HansenRBCModel with policy functions
- `k_path`: Simulated capital path
- `z_path`: Technology shock path
- `state_indices`: Discrete state indices for each period

# Returns
- `RBCAggregateVars`: All aggregate variables
"""
function compute_aggregates(model::HansenRBCModel, k_path::Vector{Float64},
    z_path::Vector{Float64}, state_indices::Vector{Int})::RBCAggregateVars
    T = length(k_path)
    params = model.params
    endo_vars = model.endogenous_vars
    k_grid = model.k_0

    # Pre-allocate aggregates
    C = zeros(T)
    H = zeros(T)
    Y = zeros(T)
    R = zeros(T)
    W = zeros(T)

    for t in 1:T
        k_t = k_path[t]
        z_t = z_path[t]
        state_idx = state_indices[t]

        # Interpolate labor policy function
        h_t = interpolate_policy(endo_vars.policy_h[:, state_idx], k_grid, k_t)

        # Compute aggregates for period t
        y_t = production(params, k_t, z_t, h_t)

        # For last period, assume steady state investment
        if t == T
            k_next = k_t  # No depreciation in final period
        else
            k_next = k_path[t+1]
        end

        c_t = y_t + (1 - params.δ) * k_t - k_next

        # Safety check for consumption
        if c_t <= 1e-10
            @warn "Non-positive consumption detected in period $t: c_t = $c_t"
            # Adjust to minimum positive consumption
            c_t = 1e-6
        end

        r_t = α * z_t * (h_t - params.h_bar)^(1 - α) * k_t^(α - 1) - params.δ
        w_t = (1 - α) * z_t * k_t^α * (h_t - params.h_bar)^(-α)

        # Store values
        C[t] = c_t
        H[t] = h_t
        Y[t] = y_t
        R[t] = r_t
        W[t] = w_t
    end

    return RBCAggregateVars(C, k_path, H, Y, z_path, R, W, T)
end

"""
    simulate_capital_path(model::HansenRBCModel, k_init::Float64, shocks::ShockSeries)::Vector{Float64}

Simulate capital path given shock series using policy functions.

# Arguments
- `model`: Solved HansenRBCModel with policy functions
- `k_init`: Initial capital level
- `shocks`: Shock series to use

# Returns
- `k_path`: Simulated capital path
"""
function simulate_capital_path(model::HansenRBCModel, k_init::Float64, shocks::ShockSeries)::Vector{Float64}
    T = shocks.T
    k_path = zeros(T)
    k_path[1] = k_init

    endo_vars = model.endogenous_vars
    k_grid = model.k_0

    for t in 1:T-1
        k_t = k_path[t]
        state_idx = shocks.state_indices[t]

        # Interpolate policy functions
        k_next = interpolate_policy(endo_vars.policy_k[:, state_idx], k_grid, k_t)

        # Safety check: ensure k_next is within reasonable bounds
        if k_next <= k_grid[1] || k_next >= k_grid[end] || !isfinite(k_next)
            @warn "Invalid capital choice detected in period $t: k_next = $k_next, using boundary"
            k_next = clamp(k_next, k_grid[1], k_grid[end])
        end

        k_path[t+1] = k_next
    end

    return k_path
end

"""
    simulate_model(model::HansenRBCModel, T::Int;
                   k_init::Union{Float64,Nothing}=nothing,
                   spec::Union{ShockSpec,Nothing}=nothing,
                   burn_in::Int=100,
                   seed::Int=1234)::RBCSimulationResult

Main simulation function for the RBC model.

# Arguments
- `model`: Solved HansenRBCModel
- `T`: Number of periods to simulate (after burn-in)
- `k_init`: Initial capital (uses steady state if nothing)
- `spec`: Shock specification (uses baseline if nothing)
- `burn_in`: Number of burn-in periods
- `seed`: Random seed

# Returns
- `RBCSimulationResult`: Complete simulation results
"""
function simulate_model(model::HansenRBCModel, T::Int;
    k_init::Union{Float64,Nothing}=nothing,
    spec::Union{ShockSpec,Nothing}=nothing,
    burn_in::Int=100,
    seed::Int=1234)::RBCSimulationResult

    # Set default shock specification (baseline stochastic process)
    if isnothing(spec)
        spec = ShockSpec(:technology, 0.0)
    end

    # Set default initial capital (steady state)
    if isnothing(k_init)
        steady_state = compute_steady_state(model.params, model.process)
        k_init = steady_state.k_star
    end

    # Generate shock series with burn-in
    total_T = T + burn_in
    shocks = generate_shock_series(model.process, total_T, spec, seed=seed)

    try
        # Simulate capital path
        k_path = simulate_capital_path(model, k_init, shocks)

        # Remove burn-in period
        k_path_sim = k_path[(burn_in+1):end]
        z_path_sim = shocks.z_series[(burn_in+1):end]
        state_indices_sim = shocks.state_indices[(burn_in+1):end]

        # Compute aggregates
        aggregates = compute_aggregates(model, k_path_sim, z_path_sim, state_indices_sim)

        # Create trimmed shock series
        trimmed_shocks = ShockSeries(z_path_sim, state_indices_sim, T, shocks.seed, shocks.spec)

        info = Dict(
            "burn_in" => burn_in,
            "k_init" => k_init,
            "T_total" => total_T,
            "T_simulated" => T,
            "shock_type" => string(spec.shock_type),
            "shock_size" => spec.shock_size
        )

        return RBCSimulationResult(
            aggregates,
            trimmed_shocks,
            "Simulation completed successfully",
            true,
            info
        )

    catch e
        return RBCSimulationResult(
            RBCAggregateVars(zeros(T), zeros(T), zeros(T), zeros(T), zeros(T), zeros(T), zeros(T), T),
            shocks,
            "Simulation failed: $(e.msg)",
            false,
            Dict("error" => e.msg)
        )
    end
end

# Convenience functions for common counterfactual analyses

"""
    compute_irf(model::HansenRBCModel, shock_size::Float64; horizon::Int=40)::RBCSimulationResult

Compute impulse response function to a one-time technology shock.

# Arguments
- `model`: Solved HansenRBCModel
- `shock_size`: Size of shock in standard deviations
- `horizon`: Number of periods to simulate

# Returns
- `RBCSimulationResult`: IRF simulation results
"""
function compute_irf(model::HansenRBCModel, shock_size::Float64; horizon::Int=40)::RBCSimulationResult
    steady_state = compute_steady_state(model.params, model.process)
    spec = ShockSpec(:technology, shock_size, timing=1, duration=1)

    return simulate_model(model, horizon, k_init=steady_state.k_star, spec=spec, burn_in=0)
end