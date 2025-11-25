using Distributions, LinearAlgebra

"""
tauchen: Discretizing AR(1) process 

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
function tauchen(ρ::Float64, σ_e::Float64, μ::Float64; N::Integer=7, m::Integer=3, taking_log::Bool=true)::Tuple{Matrix{Float64}, Vector{Float64}}
    σ_x = sqrt(σ_e^2 / (1 - ρ^2))
    zmin, zmax = -m * σ_x, m * σ_x
    z = range(zmin, zmax, length=N) |> collect
    Δ = z[2] - z[1]
    P = zeros(N, N)
    # the tauchen routine
    for i in 1:N
        for j in 1:N
            if j == 1
                P[i, j] = cdf(Normal(0, 1), (z[1] + Δ / 2 - ρ * z[i]) / σ_e)
            elseif j == N
                P[i, j] = 1 - cdf(Normal(0, 1), (z[N] - Δ / 2 - ρ * z[i]) / σ_e)
            else
                upper = (z[j] + Δ / 2 - ρ * z[i]) / σ_e
                lower = (z[j] - Δ / 2 - ρ * z[i]) / σ_e
                P[i, j] = cdf(Normal(0, 1), upper) - cdf(Normal(0, 1), lower)
            end
        end
    end
    # picked from QuantEcon.jl
    z = z .+ μ / (1 - ρ) # Adjust for mean
    if taking_log
        z = exp.(z)
    end
    P = P ./ sum(P, dims=2) # Normalize rows to sum to 1
    return P, z
end
