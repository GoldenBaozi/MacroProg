using Revise
using Plots
using Random
using MacroProg
using Interpolations
using NLsolve
using Optim

println("=== Debug RBC Model Bellman Equation Issues ===")

# Set up model parameters
params = RBCParams(
    0.98,   # β: discount factor
    0.36,   # α: capital share
    0.10,   # δ: depreciation rate
    1.0,    # B: leisure preference parameter
    0.2     # h_bar: minimum labor supply
)

# Set up stochastic process for technology shocks
process = AROneProcess(
    0.75,    # ρ: autocorrelation
    0.07,  # σ: std deviation of shocks
    0.0;    # μ: mean
    N=7,     # number of discrete states
    m=3,     # standard deviations to cover
    taking_log=true
)

# Set up capital grid
k_0 = collect(0.4:0.2:16)

# Create and solve the model
println("Creating and solving the model...")
model = HansenRBCModel(params, process, k_0)
println("Solving model using time iteration...")
solve_model!(model; method="TI", tol=1e-6, max_iter=200)

if model.solving_result.status.status == 0
    println("✓ Model solved successfully!")
    println("✓ Converged in $(model.solving_result.n_loops) iterations")
else
    println("✗ Model solving failed: $(model.solving_result.status.message)")
end

# Extract results
v_func = model.endogenous_vars.v_func
policy_k = model.endogenous_vars.policy_k
policy_h = model.endogenous_vars.policy_h
policy_c = model.endogenous_vars.policy_c

println("\n=== Analyzing Results ===")

# Check monotonicity of value function
println("Checking value function monotonicity...")
for j in 1:process.N
    monotonic = all(diff(v_func[:, j]) .>= -1e-10)  # Allow small numerical errors
    if monotonic
        println("✓ State $j: Value function is monotonic increasing")
    else
        println("✗ State $j: Value function is NOT monotonic increasing")
        # Find where it decreases
        decreases = findall(diff(v_func[:, j]) .< -1e-10)
        println("  Decreases at indices: $decreases")
    end
end

# Check labor policy monotonicity
println("\nChecking labor policy monotonicity...")
for j in 1:process.N
    # In RBC models, labor should be non-monotonic in capital
    # Let's check the pattern
    h_values = policy_h[:, j]
    max_idx = argmax(h_values)
    min_idx = argmin(h_values)

    println("State $j:")
    println("  Labor range: [$(round(minimum(h_values), digits=3)), $(round(maximum(h_values), digits=3))]")
    println("  Min at k=$(round(k_0[min_idx], digits=2)), Max at k=$(round(k_0[max_idx], digits=2))")

    # Check if it's monotonic (which would be wrong)
    monotonic_increasing = all(diff(h_values) .>= -1e-10)
    monotonic_decreasing = all(diff(h_values) .<= 1e-10)

    if monotonic_increasing
        println("  ✗ WARNING: Labor is monotonic increasing in capital (theoretically wrong)")
    elseif monotonic_decreasing
        println("  ✗ WARNING: Labor is monotonic decreasing in capital (theoretically wrong)")
    else
        println("  ✓ Labor is non-monotonic in capital (theoretically correct)")
    end
end

# Plot results for visual inspection
p1 = plot(title="Value Function by State", xlabel="Capital", ylabel="Value")
for j in 1:process.N
    plot!(p1, k_0, v_func[:, j], label="State $j", linewidth=2)
end

p2 = plot(title="Labor Policy by State", xlabel="Capital", ylabel="Labor Hours")
for j in 1:process.N
    plot!(p2, k_0, policy_h[:, j], label="State $j", linewidth=2)
end

p3 = plot(title="Capital Policy by State", xlabel="Current Capital", ylabel="Next Period Capital")
for j in 1:process.N
    plot!(p3, k_0, policy_k[:, j], label="State $j", linewidth=2)
end

display(p1)
display(p2)
display(p3)

println("\n=== Testing Individual Functions ===")

# Test the Bellman equation components
k_test = k_0[10]  # Middle of the grid
z_test = process.states[4]  # Middle state
println("Testing at k=$(round(k_test, digits=2)), z=$(round(z_test, digits=3))")

# Test utility gradient
du_dk_next, du_dh = grad_utility(params, k_test, k_test, 0.5, z_test)
println("Utility gradients: du/dk_next = $(round(du_dk_next, digits=4)), du/dh = $(round(du_dh, digits=4))")

# Test labor FOC
foc_val = foc_labor(params, k_test, k_test, 0.5, z_test)
println("Labor FOC value: $(round(foc_val, digits=4))")

# Test optimal choice
best_k_next, best_h = vfi_optimal_choice(params, k_test, z_test, process, k_0, zeros(size(v_func)))
println("Optimal choice: k_next=$(round(best_k_next, digits=3)), h=$(round(best_h, digits=3))")

# Verify resource constraint
y = production(params, k_test, z_test, best_h)
c = y + (1 - params.δ) * k_test - best_k_next
println("Resource constraint check:")
println("  Production y = $(round(y, digits=3))")
println("  Consumption c = $(round(c, digits=3))")
println("  Next period k = $(round(best_k_next, digits=3))")
println("  Resource balance: $(round(y + (1 - params.δ) * k_test - c - best_k_next, digits=8))")