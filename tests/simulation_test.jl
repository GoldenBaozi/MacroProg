using Revise
using Plots
using Random
using MacroProg
using Plots
using Interpolations
using NLsolve
using Optim

println("=== Testing RBC Model Simulation Extensions ===")

# 1. Set up model parameters
println("\n1. Setting up model parameters...")
params = RBCParams(
    0.98,   # β: discount factor
    0.36,   # α: capital share
    0.10,   # δ: depreciation rate
    1.0,    # B: leisure preference parameter
    0.2     # h_bar: minimum labor supply
)

# 2. Set up stochastic process for technology shocks
println("2. Setting up AR(1) technology process...")
process = AROneProcess(
    0.75,    # ρ: autocorrelation
    0.07,  # σ: std deviation of shocks
    0.0;    # μ: mean
    N=7,     # number of discrete states
    m=3,     # standard deviations to cover
    taking_log=true
)

# 3. Set up capital grid
println("3. Setting up capital grid...")
k_0 = collect(0.4:0.2:16)

# 4. Create and solve the model
println("4. Crating and solving the model...")
model = HansenRBCModel(params, process, k_0)
println("   Solving model using time iteration...")
solve_model!(model; method="TI", tol=1e-6, max_iter=200)

if model.solving_result.status.status == 0
    println("   ✓ Model solved successfully!")
    println("   ✓ Converged in $(model.solving_result.n_loops) iterations")
    println("   ✓ Time used: $(round(model.solving_result.time_used, digits=3)) seconds")
else
    println("   ✗ Model solving failed: $(model.solving_result.status.message)")
    # exit()
end

v_func = model.endogenous_vars.v_func
policy_k = model.endogenous_vars.policy_k
policy_h = model.endogenous_vars.policy_h
policy_c = model.endogenous_vars.policy_c
# plot value function of diffeerent states
p = plot()
for j in 1:process.N
    plot!(p, k_0, v_func[:, j], label="State $(j)", linewidth=2)
end
title!(p, "Value Function")
display(p)

p_k = plot()
for j in 1:process.N
    plot!(p_k, k_0, policy_k[:, j], label="State $(j)", linewidth=2)
end
display(p_k)

p_h = plot()
for j in 1:process.N
    plot!(p_h, k_0, policy_h[:, j], label="State $(j)", linewidth=2)
end
title!(p_h, "Labor Policy Function")
display(p_h)

p_c = plot()
for j in 1:process.N
    plot!(p_c, k_0, policy_c[:, j], label="State $(j)", linewidth=2)
end
display(p_c)

solve_model!(model; method="VFI", tol=1e-6, max_iter=1000)
if model.solving_result.status.status == 0
    println("   ✓ Model solved successfully using VFI!")
    println("   ✓ Converged in $(model.solving_result.n_loops) iterations")
    println("   ✓ Time used: $(round(model.solving_result.time_used, digits=3)) seconds")
else
    println("   ✗ Model solving failed using VFI: $(model.solving_result.status.message)")
    # exit()
end
v_func_VFI = model.endogenous_vars.v_func
policy_k_VFI = model.endogenous_vars.policy_k
policy_h_VFI = model.endogenous_vars.policy_h
policy_c_VFI = model.endogenous_vars.policy_c

# compare TI and VFI results
p_diff = plot()
for j in 1:process.N
    plot!(p_diff, k_0, abs.(policy_h[:, j] - policy_h_VFI[:, j]), label="State $(j)", linewidth=2)
end
title!(p_diff, "Labor Policy Function Difference (TI vs VFI)")
display(p_diff)

p_k_diff = plot()
for j in 1:process.N
    plot!(p_k_diff, k_0, abs.(policy_k[:, j] - policy_k_VFI[:, j]), label="State $(j)", linewidth=2)
end
title!(p_k_diff, "Capital Policy Function Difference (TI vs VFI)")
display(p_k_diff)

p_c_diff = plot()
for j in 1:process.N
    plot!(p_c_diff, k_0, abs.(policy_c[:, j] - policy_c_VFI[:, j]), label="State $(j)", linewidth=2)
end
title!(p_c_diff, "Consumption Policy Function Difference (TI vs VFI)")
display(p_c_diff)

# # 5. Compute steady state
# println("\n5. Computing steady state...")
# steady_state = compute_steady_state(params, process)
# println("   Steady state capital: $(round(steady_state.k_star, digits=3))")
# println("   Steady state labor: $(round(steady_state.h_star, digits=3))")
# println("   Steady state consumption: $(round(steady_state.c_star, digits=3))")
# println("   Steady state output: $(round(steady_state.y_star, digits=3))")
# println("   Steady state technology: $(round(steady_state.z_bar, digits=3))")

# # 6. Test baseline simulation
# println("\n6. Testing baseline simulation...")
# T = 200
# baseline_result = simulate_model(model, T, burn_in=50, seed=1234)

# if baseline_result.converged
#     println("   ✓ Baseline simulation completed successfully!")
#     println("   ✓ Simulated $(baseline_result.aggregates.T) periods")
#     println("   ✓ Average output: $(round(mean(baseline_result.aggregates.Y), digits=3))")
#     println("   ✓ Average consumption: $(round(mean(baseline_result.aggregates.C), digits=3))")
# else
#     println("   ✗ Baseline simulation failed: $(baseline_result.status)")
# end

# # 7. Test impulse response function (positive technology shock)
# println("\n7. Testing impulse response function (+1σ technology shock)...")
# irf_pos = compute_irf(model, 1.0, horizon=40)

# if irf_pos.converged
#     println("   ✓ Positive IRF computed successfully!")
#     println("   ✓ Peak output response: $(round(maximum(irf_pos.aggregates.Y) / steady_state.y_star - 1, digits=3))%")
#     println("   ✓ Peak consumption response: $(round(maximum(irf_pos.aggregates.C) / steady_state.c_star - 1, digits=3))%")
# else
#     println("   ✗ Positive IRF failed: $(irf_pos.status)")
# end

# # 8. Test impulse response function (negative technology shock)
# println("\n8. Testing impulse response function (-1σ technology shock)...")
# irf_neg = compute_irf(model, -1.0, horizon=40)

# if irf_neg.converged
#     println("   ✓ Negative IRF computed successfully!")
#     println("   ✓ Trough output response: $(round(minimum(irf_neg.aggregates.Y) / steady_state.y_star - 1, digits=3))%")
#     println("   ✓ Trough consumption response: $(round(minimum(irf_neg.aggregates.C) / steady_state.c_star - 1, digits=3))%")
# else
#     println("   ✗ Negative IRF failed: $(irf_neg.status)")
# end

# # 9. Test custom shock specification
# println("\n9. Testing custom shock specification...")
# custom_shocks = randn(T) .* process.σ .+ process.μ
# custom_shocks = exp.(custom_shocks)  # Convert to log level
# custom_spec = ShockSpec(:custom, 0.0, custom_series=custom_shocks)
# custom_result = simulate_model(model, T, spec=custom_spec, burn_in=0)

# if custom_result.converged
#     println("   ✓ Custom shock simulation completed successfully!")
#     println("   ✓ Custom shock variance: $(round(var(custom_shocks), digits=4))")
# else
#     println("   ✗ Custom shock simulation failed: $(custom_result.status)")
# end

# # 10. Test non-steady state initialization
# println("\n10. Testing non-steady state initialization...")
# k_init_high = steady_state.k_star * 1.5  # Start with 50% more capital
# nonss_result = simulate_model(model, T, k_init=k_init_high, burn_in=0)

# if nonss_result.converged
#     println("   ✓ Non-steady state simulation completed successfully!")
#     initial_capital_ratio = nonss_result.aggregates.K[1] / steady_state.k_star
#     println("   ✓ Initial capital/SS ratio: $(round(initial_capital_ratio, digits=2))")
#     final_capital_ratio = nonss_result.aggregates.K[end] / steady_state.k_star
#     println("   ✓ Final capital/SS ratio: $(round(final_capital_ratio, digits=2))")
# else
#     println("   ✗ Non-steady state simulation failed: $(nonss_result.status)")
# end

# # 11. Visualize results
# println("\n11. Creating visualizations...")

# # Plot impulse responses
# p1 = plot(1:40, irf_pos.aggregates.Y ./ steady_state.y_star .- 1,
#           label="Output (+1σ)", linewidth=2, color=:blue)
# plot!(p1, 1:40, irf_neg.aggregates.Y ./ steady_state.y_star .- 1,
#        label="Output (-1σ)", linewidth=2, color=:red)
# plot!(p1, 1:40, irf_pos.aggregates.C ./ steady_state.c_star .- 1,
#        label="Consumption (+1σ)", linewidth=2, color=:blue, linestyle=:dash)
# plot!(p1, 1:40, irf_neg.aggregates.C ./ steady_state.c_star .- 1,
#        label="Consumption (-1σ)", linewidth=2, color=:red, linestyle=:dash)
# xlabel!("Periods after shock")
# ylabel!("Percentage deviation from steady state")
# title!("Impulse Response Functions")
# hline!([0], color=:black, linestyle=:dot, label="")

# # Plot baseline simulation time series
# p2 = plot(1:T, baseline_result.aggregates.Y, label="Output", linewidth=1.5)
# plot!(p2, 1:T, baseline_result.aggregates.C, label="Consumption", linewidth=1.5)
# plot!(p2, 1:T, baseline_result.aggregates.K, label="Capital", linewidth=1.5)
# xlabel!("Time")
# ylabel!("Level")
# title!("Baseline Simulation Time Series")
# legend!(:topleft)

# # Plot technology shocks
# p3 = plot(1:T, baseline_result.aggregates.Z, label="Technology Shocks",
#           linewidth=1.5, color=:green)
# xlabel!("Time")
# ylabel!("Technology Level")
# title!("Realized Technology Shocks")
# legend!(:topleft)

# # Plot convergence from non-steady state
# p4 = plot(1:T, nonss_result.aggregates.K ./ steady_state.k_star,
#           label="Capital/SS", linewidth=2, color=:purple)
# xlabel!("Time")
# ylabel!("Ratio to Steady State")
# title!("Convergence from Non-Steady State (Initial K = 1.5×SS)")
# hline!([1.0], color=:black, linestyle=:dash, label="Steady State")
# legend!(:topright)

# display(p1)
# display(p2)
# display(p3)
# display(p4)

# # 12. Summary statistics
# println("\n12. Summary Statistics")
# println("=" ^ 50)

# if baseline_result.converged
#     println("BASELINE SIMULATION ($(T) periods):")
#     println("  Output:     μ = $(round(mean(baseline_result.aggregates.Y), digits=3)), σ = $(round(std(baseline_result.aggregates.Y), digits=3))")
#     println("  Consumption: μ = $(round(mean(baseline_result.aggregates.C), digits=3)), σ = $(round(std(baseline_result.aggregates.C), digits=3))")
#     println("  Hours:      μ = $(round(mean(baseline_result.aggregates.H), digits=3)), σ = $(round(std(baseline_result.aggregates.H), digits=3))")
#     println("  Investment: μ = $(round(mean(baseline_result.aggregates.K[2:end] - baseline_result.aggregates.K[1:end-1] + params.δ * baseline_result.aggregates.K[1:end-1]), digits=3))")
#     println("  Tech Z:     μ = $(round(mean(baseline_result.aggregates.Z), digits=3)), σ = $(round(std(baseline_result.aggregates.Z), digits=3))")
# end

# println("\nIMPULSE RESPONSES (Peak Effects):")
# println("  +1σ Tech Shock:")
# println("    Output:      $(round(maximum(irf_pos.aggregates.Y) / steady_state.y_star - 1, digits=3))%")
# println("    Consumption: $(round(maximum(irf_pos.aggregates.C) / steady_state.c_star - 1, digits=3))%")
# println("    Hours:       $(round(maximum(irf_pos.aggregates.H) / steady_state.h_star - 1, digits=3))%")

# println("  -1σ Tech Shock:")
# println("    Output:      $(round(minimum(irf_neg.aggregates.Y) / steady_state.y_star - 1, digits=3))%")
# println("    Consumption: $(round(minimum(irf_neg.aggregates.C) / steady_state.c_star - 1, digits=3))%")
# println("    Hours:       $(round(minimum(irf_neg.aggregates.H) / steady_state.h_star - 1, digits=3))%")

# println("\n" * "=" * 50)
# println("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
# println("✓ RBC model simulation extensions are working correctly!")
# println("=" * 50)