module MacroProg
# __precompile__(false)


include("tauchen.jl")
include("AbstractTypes.jl")
include("HansenRBC.jl")
include("coleman_iteration.jl")

# Core exports
export tauchen

# Abstract types
export AbstractModel, AbstractProcess, AbstractParams, AbstractVariables, AbstractSolvingResult
export AbstractSteadyState, AbstractAggregates, AbstractShockSeries, AbstractSimulationResult

# RBC Model types
export RBCParams, AROneProcess, HansenRBCModel, RBCEndogenousVar, VFIResult, SolvingStatus

# Simulation types
export RBCSteadyState, ShockSpec, RBCAggregateVars, ShockSeries, RBCSimulationResult

# Core functions
export solve_model!, production, utility, pfi_optimal_choice, time_iteration

end # module MacroProg
