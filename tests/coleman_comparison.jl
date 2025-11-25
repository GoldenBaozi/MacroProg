using Revise
using MacroProg
using BenchmarkTools
using Plots

"""
Test script to compare Value Function Iteration vs Coleman Policy Iteration
for the Hansen RBC model.
"""

println("=== RBC Model Solver Comparison ===")
println("Comparing VFI vs Coleman Policy Iteration\n")

# 1. Set up the model
println("1. Setting up the model...")
params = RBCParams(0.98, 0.36, 0.1, 1.0, 0.33)
process = AROneProcess(0.9, 0.01, 0.0; N=7, m=3)
k_grid = collect(0.4:0.2:16)

println("Parameters:")
println("  β = $(params.β), α = $(params.α), δ = $(params.δ)")
println("  Process: ρ = $(process.ρ), σ = $(process.σ), N = $(process.N)")
println("  Capital grid: $(length(k_grid)) points from $(k_grid[1]) to $(k_grid[end])")

model = HansenRBCModel(params, process, k_grid)
println("✓ Model created\n")

# 2. Solve with Value Function Iteration
println("2. Solving with Value Function Iteration...")
vfi_time = @elapsed begin
    solve_model!(model; tol=1e-6, max_iter=1000)
end
vfi_result = model.solving_result

println("VFI Results:")
println("  Status: $(vfi_result.status.message)")
println("  Iterations: $(vfi_result.n_loops)")
println("  Time: $(round(vfi_time, digits=3)) seconds")
println("  Converged: $(vfi_result.status.status == 0)")

# 3. Solve with Coleman Policy Iteration
println("\n3. Solving with Coleman Policy Iteration...")
coleman_time = @elapsed begin
    coleman_result = coleman_policy_iteration(model; tol=1e-6, max_iter=100)
end

println("Coleman Results:")
println("  Status: $(coleman_result.status.message)")
println("  Iterations: $(coleman_result.n_loops)")
println("  Time: $(round(coleman_time, digits=3)) seconds")
println("  Converged: $(coleman_result.status.status == 0)")

# 4. Compare policy functions
println("\n4. Comparing policy functions...")

# Get policies from both methods
vfi_k_policy = vfi_result.endo_vars.policy_k
vfi_h_policy = vfi_result.endo_vars.policy_h
coleman_k_policy = coleman_result.endo_vars.policy_k
coleman_h_policy = coleman_result.endo_vars.policy_h

# Compute differences
k_policy_diff = maximum(abs.(vfi_k_policy .- coleman_k_policy))
h_policy_diff = maximum(abs.(vfi_h_policy .- coleman_h_policy))

println("Maximum policy differences:")
println("  Capital policy: $(round(k_policy_diff, digits=6))")
println("  Labor policy: $(round(h_policy_diff, digits=6))")

# 5. Performance benchmarking
println("\n5. Performance benchmarking...")

println("VFI benchmark:")
@benchmark solve_model!(deepcopy(model); tol=1e-6, max_iter=1000)

println("\nColeman benchmark:")
@benchmark coleman_policy_iteration(deepcopy(model); tol=1e-6, max_iter=100)

# 6. Visual comparison
println("\n6. Creating visual comparison...")

# Choose a state (e.g., middle technology shock state)
state_idx = div(process.N, 2) + 1
z_val = process.states[state_idx]

# Plot capital policies
p1 = plot(k_grid, vfi_k_policy[:, state_idx], label="VFI",
          title="Capital Policy Functions (z = $(round(z_val, digits=3)))",
          xlabel="Current Capital (k)", ylabel="Next Period Capital (k')",
          linewidth=2, legend=:topleft)
plot!(p1, k_grid, coleman_k_policy[:, state_idx], label="Coleman",
      linewidth=2, linestyle=:dash)
plot!(p1, k_grid, k_grid, label="45° line", linestyle=:dot, color=:black)

# Plot labor policies
p2 = plot(k_grid, vfi_h_policy[:, state_idx], label="VFI",
          title="Labor Policy Functions (z = $(round(z_val, digits=3)))",
          xlabel="Current Capital (k)", ylabel="Labor (h)",
          linewidth=2, legend=:topleft)
plot!(p2, k_grid, coleman_h_policy[:, state_idx], label="Coleman",
      linewidth=2, linestyle=:dash)

# Plot policy differences
p3 = plot(k_grid, vfi_k_policy[:, state_idx] .- coleman_k_policy[:, state_idx],
          label="Capital", title="Policy Function Differences",
          xlabel="Current Capital (k)", ylabel="Difference (VFI - Coleman)",
          linewidth=2, legend=:bottomright)
plot!(p3, k_grid, vfi_h_policy[:, state_idx] .- coleman_h_policy[:, state_idx],
      label="Labor", linewidth=2)

# Combine plots
combined_plot = plot(p1, p2, p3, layout=(1,3), size=(1200, 400),
                    title="VFI vs Coleman Policy Iteration Comparison")

# Save the plot
savefig(combined_plot, "vfi_vs_coleman_comparison.png")
println("✓ Comparison plot saved as 'vfi_vs_coleman_comparison.png'")

# 7. Test with simulation
println("\n7. Testing simulation performance...")

# Test steady state computation
println("Computing steady state...")
steady_state = compute_steady_state(params, process)
println("Steady state: k* = $(round(steady_state.k_star, digits=3)), h* = $(round(steady_state.h_star, digits=3))")

# Test simulation with both methods
println("\nTesting simulation with VFI policies...")
vfi_sim_time = @elapsed begin
    vfi_sim_result = simulate_model(model, 100, k_init=steady_state.k_star, seed=42)
end

println("Testing simulation with Coleman policies...")
# Temporarily replace policies for simulation test
original_endogenous = model.endogenous_vars
model.endogenous_vars = coleman_result.endo_vars
coleman_sim_time = @elapsed begin
    coleman_sim_result = simulate_model(model, 100, k_init=steady_state.k_star, seed=42)
end
model.endogenous_vars = original_endogenous  # Restore original

println("Simulation times:")
println("  VFI policies: $(round(vfi_sim_time, digits=3)) seconds")
println("  Coleman policies: $(round(coleman_sim_time, digits=3)) seconds")

# Compare simulation outputs
vfi_output = vfi_sim_result.aggregates
coleman_output = coleman_sim_result.aggregates

println("Simulation comparison (final period):")
println("  VFI - C: $(round(vfi_output.C[end], digits=4)), K: $(round(vfi_output.K[end], digits=4))")
println("  Coleman - C: $(round(coleman_output.C[end], digits=4)), K: $(round(coleman_output.K[end], digits=4))")

# 8. Summary
println("\n" * "="^60)
println("SUMMARY")
println("="^60)

println("\nSpeed Comparison:")
println("  VFI: $(round(vfi_time, digits=3))s ($(vfi_result.n_loops) iterations)")
println("  Coleman: $(round(coleman_time, digits=3))s ($(coleman_result.n_loops) iterations)")

speedup = vfi_time / coleman_time
println("  Speedup: $(round(speedup, digits=2))x")

println("\nAccuracy:")
println("  Max policy difference (k): $(round(k_policy_diff, digits=6))")
println("  Max policy difference (h): $(round(h_policy_diff, digits=6))")

if k_policy_diff < 1e-4 && h_policy_diff < 1e-4
    println("  ✓ High accuracy agreement")
elseif k_policy_diff < 1e-3 && h_policy_diff < 1e-3
    println("  ✓ Good accuracy agreement")
else
    println("  ⚠ Significant policy differences")
end

println("\nRecommendations:")
if speedup > 2.0 && k_policy_diff < 1e-3
    println("  ✓ Coleman iteration is recommended - significantly faster with good accuracy")
elseif speedup > 1.5
    println("  → Coleman iteration offers moderate speed benefits")
else
    println("  → VFI may be preferred for this model configuration")
end

println("\nFiles generated:")
println("  - vfi_vs_coleman_comparison.png: Policy function comparison plots")
println("  - coleman_comparison_results.txt: Detailed numerical results")

# 9. Save detailed results
results_file = open("coleman_comparison_results.txt", "w")
write(results_file, "RBC Model Solver Comparison Results\n")
write(results_file, "====================================\n\n")
write(results_file, "Model Parameters:\n")
write(results_file, "  β = $(params.β), α = $(params.α), δ = $(params.δ), B = $(params.B), h̄ = $(params.h_bar)\n")
write(results_file, "  Process: ρ = $(process.ρ), σ = $(process.σ), N = $(process.N)\n\n")

write(results_file, "Value Function Iteration:\n")
write(results_file, "  Status: $(vfi_result.status.message)\n")
write(results_file, "  Iterations: $(vfi_result.n_loops)\n")
write(results_file, "  Time: $(vfi_time) seconds\n")
write(results_file, "  Converged: $(vfi_result.status.status == 0)\n\n")

write(results_file, "Coleman Policy Iteration:\n")
write(results_file, "  Status: $(coleman_result.status.message)\n")
write(results_file, "  Iterations: $(coleman_result.n_loops)\n")
write(results_file, "  Time: $(coleman_time) seconds\n")
write(results_file, "  Converged: $(coleman_result.status.status == 0)\n\n")

write(results_file, "Policy Differences:\n")
write(results_file, "  Maximum capital policy difference: $k_policy_diff\n")
write(results_file, "  Maximum labor policy difference: $h_policy_diff\n")
write(results_file, "  Speedup factor: $(vfi_time/coleman_time)\n")
close(results_file)

println("\n✓ All tests completed successfully!")