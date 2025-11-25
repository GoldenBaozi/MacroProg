using LinearAlgebra, Plots, Interpolations, Optim

# create k_0 that: start from 0.4, step 0.1, end 16
k_0 = 0.4:0.1:16
k_0 = collect(k_0)
n_grid = length(k_0)
kt_11 = copy(k_0)
kt_12 = copy(k_0)
β = 0.98
δ = 0.1
θ = 0.36

A1 = 1.75
p_11 = 0.9
p_12 = 1 - p_11
p_21 = 0.4
p_22 = 1 - p_21
A2 = 0.75
numits = 1000
tolerance = 1e-6

# build vlast_1 as a 40 length ones vector times 20
# change vlast_1 to vector instead of matrix

vlast_1 = ones(n_grid) * 20
vlast_2 = ones(n_grid) * 20

# Function to compute optimal choice using interpolation and optimization
function compute_optimal_choice(kt::Float64, At, p1, p2, vlast_1_interp, vlast_2_interp, k_0, β, δ, θ)
    # Compute maximum feasible capital (total output minus depreciation)
    total_resources = At * kt^θ + (1 - δ) * kt

    # Ensure positive consumption by setting upper bound below total resources
    # Leave small amount for consumption to avoid log(0)
    k_next_max = total_resources * 0.999  # Leave at least 0.1% for consumption

    # Define objective function for given k_next
    function objective(k_next)
        consumption = total_resources - k_next

        # Ensure strictly positive consumption
        if consumption <= 1e-10 || k_next <= 0
            return -Inf
        end

        # Interpolate expected future value
        exp_future_value = p1 * vlast_1_interp(k_next) + p2 * vlast_2_interp(k_next)

        # Current period utility plus expected discounted future value
        current_utility = log(consumption)
        return current_utility + β * exp_future_value
    end

    # Use numerical optimization to find maximum
    # Lower bound should be strictly positive to maintain capital positivity
    result = optimize(f=k_next -> -objective(k_next), 0.1, k_next_max, GoldenSection())

    if Optim.converged(result)
        optimal_k_next = Optim.minimizer(result)
        max_val = objective(optimal_k_next)
        return max_val, optimal_k_next
    else
        # Fallback to grid search if optimization fails
        v_next = zeros(length(k_0))
        for l in 1:length(k_0)
            if k_0[l] <= k_next_max
                v_next[l] = log(At * kt^θ + (1 - δ) * kt - k_0[l]) + β * (p1 * vlast_1_interp(k_0[l]) + p2 * vlast_2_interp(k_0[l]))
            end
        end
        max_val, max_idx = findmax(v_next)
        return max_val, k_0[max_idx]
    end
end

k = 1
converged = false

while !converged && k <= numits
    v1 = zeros(n_grid)
    v2 = zeros(n_grid)

    # Create interpolation objects for current value functions
    vlast_1_interp = linear_interpolation(k_0, vlast_1, extrapolation_bc=Line())
    vlast_2_interp = linear_interpolation(k_0, vlast_2, extrapolation_bc=Line())

    for j in 1:n_grid
        kt = k_0[j]

        # State 1 optimization
        At = A1
        p1 = p_11
        p2 = p_12
        v1[j], kt_11[j] = compute_optimal_choice(kt, At, p1, p2, vlast_1_interp, vlast_2_interp, k_0, β, δ, θ)

        # State 2 optimization
        At = A2
        p1 = p_21
        p2 = p_22
        v2[j], kt_12[j] = compute_optimal_choice(kt, At, p1, p2, vlast_1_interp, vlast_2_interp, k_0, β, δ, θ)
    end

    # Check convergence
    max_diff_1 = maximum(abs.(v1 - vlast_1))
    max_diff_2 = maximum(abs.(v2 - vlast_2))
    max_diff = max(max_diff_1, max_diff_2)

    if k / 100 == round(k / 100) || max_diff < tolerance
        println("Iteration ", k)
        println("Max difference State 1: ", max_diff_1)
        println("Max difference State 2: ", max_diff_2)
    end

    converged = max_diff < tolerance

    if converged
        println("Converged after ", k, " iterations with tolerance ", tolerance)
    end

    vlast_1 = v1
    vlast_2 = v2
    k += 1
end

if !converged
    println("Maximum iterations (", numits, ") reached without convergence")
end

plt_policy = plot(k_0, kt_11, label="Policy Function State 1", xlabel="Capital Today", ylabel="Capital Tomorrow", title="Policy Functions", color=:blue, linestyle=:dash)
plot!(plt_policy, k_0, kt_12, label="Policy Function State 2", color=:orange, linestyle=:dot)
display(plt_policy)

plt_value = plot(k_0, vlast_1, label="Value Function State 1", xlabel="Capital", ylabel="Value", title="Value Functions", color=:green, linestyle=:solid)
plot!(plt_value, k_0, vlast_2, label="Value Function State 2", color=:red, linestyle=:dashdot)
display(plt_value)


