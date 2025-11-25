import QuantEcon
using Test
using MacroProg: tauchen
using Revise

MC = QuantEcon.tauchen(7, 0.5, 0.1, 0.5, 3.0)
P_expected, z_expected = tauchen(0.5, 0.1, 0, N=7)

# compare MC.p and P_expected
@test isapprox(MC.p, P_expected; atol=1e-10)
# compare MC.state_values and z_expected
@test isapprox(MC.state_values, z_expected; atol=1e-10)

sum(P_expected, dims=2)
sum(MC.p, dims=2)