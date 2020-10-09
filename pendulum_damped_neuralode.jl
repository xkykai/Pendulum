using DifferentialEquations, Plots
using DiffEqFlux, Optim
using Flux
using BlackBoxOptim
using LaTeXStrings
using Statistics

function damped_pendulum!(du, u, p, t)
    g,L,b = p
    θ, dθ = u
    du[1] = dθ
    du[2] = -g/L * sin(θ) - b * dθ
end

u₀ = [0.5, 0]
tspan_train = (0.0, 0.5)
p_train = [9.81, 0.1, 1]
Δt_train = 0.05

prob = ODEProblem(damped_pendulum!, u₀, tspan_train, p_train)

sol = solve(prob, saveat=Δt_train)

scatter(sol, vars = (1), label = "displacement")
scatter(sol, vars = (2), label = "velocity")

dataset_train = Array(sol)
t_train = Array(sol.t)

function feature_scaling(x, mean, std)
    (x .- mean) ./ std
end

function feature_scaling_dataset(dataset)
    dataset_scaled = similar(dataset)
    θ = dataset[1,:]
    mean_θ = mean(θ)
    std_θ = std(θ)
    dataset_scaled[1,:] = feature_scaling(θ, mean_θ, std_θ)

    dθ = dataset[2,:]
    mean_dθ = mean(dθ)
    std_dθ = std(dθ)
    dataset_scaled[2,:] = feature_scaling(dθ, mean_dθ, std_dθ)
    return dataset_scaled, mean_θ, std_θ, mean_dθ, std_dθ
end

dataset_train_scaled, mean_train_θ, std_train_θ, mean_train_dθ, std_train_dθ = feature_scaling_dataset(dataset_train)

function total_energy(dθ, θ, p)
    g = p[1]
    r = p[2]
    0.5 .* r .^2 .* (dθ .^2) .+ g .* r .* (1 .- cos.(θ))
end

function energy_change(E)
    E[2:end] - E[1:end-1]
end

model = FastChain(FastDense(2, 50, tanh), FastDense(50, 2))
prob_NN = NeuralODE(model, tspan_train, Tsit5(), saveat =Δt_train)
# plot(prob_NN(u₀))

function predict(p)
    Array(prob_NN(u₀, p))
end

# function loss_n_ode(p)
#     pred = predict(p)
#     E = total_energy(pred[2,:], pred[1,:], p_train)
#     ΔE = energy_change(E)
#     E_penalty = 0
#     for i in 1:length(ΔE)
#         if (ΔE[i] > 0)
#             E_penalty += ΔE[i] * 10
#         end
#     end
#     loss_dθ_reg = Flux.mse(dataset_train_dθ_reg, feature_scaling(pred[2,:], mean_train_dθ, std_train_dθ))
#     loss_θ_reg = Flux.mse(dataset_train_θ_reg, feature_scaling(pred[1,:], mean_train_θ, std_train_θ))
#     # sum(abs2, dataset .- pred[1:2, :]), pred
#     mean(loss_θ_reg + loss_dθ_reg) + E_penalty, pred
# end

function loss_n_ode(p)
    pred = predict(p)
    loss_θ = Flux.mse(dataset_train_scaled[1,:], feature_scaling(pred[1,:], mean_train_θ, std_train_θ))
    loss_dθ = Flux.mse(dataset_train_scaled[2,:], feature_scaling(pred[2,:], mean_train_dθ, std_train_dθ))
    mean(loss_θ + loss_dθ), pred
end

cb = function (p,l,pred)
    println(l)
    false
end

cb_plot = function (p,l,pred)
    println(l)
    fig = plot(t_train,dataset_train[1,:], ylims=(-1,1), xlims=tspan_train)
    plot!(fig, t_train, pred[1,:])
    display(fig)
    false
end

cb_plot_vel = function (p,l,pred)
    println(l)
    fig = plot(t_train,dataset_train[2,:], ylims=(-25,25), xlims=tspan_train)
    plot!(fig, t_train, pred[2,:])
    display(fig)
    false
end


println("Starting Gradient Descent")
res = DiffEqFlux.sciml_train(loss_n_ode, prob_NN.p, Descent(), maxiters=5000)
println("Starting ADAM")
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, ADAM(0.01), maxiters = 5000)
println("Starting BFGS")
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, BFGS(), cb = cb_plot)
println("Starting LBFGS")
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, LBFGS(), cb = cb_plot)
# println("Starting NADAM")
# res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, NADAM(), maxiters = 5000)

loss_n_ode(res.minimizer)[1]

tspan_test = (0., 2.)
# u₀_test = [0.1,0]
u₀_test = [0.5,0]

prob_test_NN = NeuralODE(model, tspan_test, Tsit5(), p=res.minimizer)
sol_test_NN = prob_test_NN(u₀_test)

prob_test = remake(prob, tspan = tspan_test, u0=u₀_test)
sol_test = solve(prob_test)

plot(sol_test_NN, vars = (1), label= "NN θ")
plot!(sol_test, vars = (1), label = "truth θ")

plot(sol_test_NN, vars = (2), label= "NN dθ")
plot!(sol_test, vars = (2), label = "truth dθ")


tspan_train = (0., 1.5)

sol = solve(prob, saveat=Δt_train, tspan = tspan_train)

dataset_train = Array(sol)
t_train = Array(sol.t)

dataset_train_scaled, mean_train_θ, std_train_θ, mean_train_dθ, std_train_dθ = feature_scaling_dataset(dataset_train)

prob_NN = NeuralODE(model, tspan_train, Tsit5(), saveat =Δt_train, p=res.minimizer)

scatter(sol, vars = (1), label = "truth θ")
scatter!(prob_NN(u₀), vars = (1), label = "NN θ")

scatter(sol, vars = (2), label = "truth dθ")
scatter!(prob_NN(u₀), vars = (2), label = "NN dθ")

loss_n_ode(res.minimizer)[1]

println("Starting ADAM")
res2 = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, ADAM(0.01), maxiters = 10000)
println("Starting BFGS")
res2 = DiffEqFlux.sciml_train(loss_n_ode, res2.minimizer, BFGS(), cb = cb)
println("Starting LBFGS")
res2 = DiffEqFlux.sciml_train(loss_n_ode, res2.minimizer, LBFGS(), cb=cb)
# println("Starting NADAM")
# res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, NADAM(), maxiters = 5000)


loss_n_ode(res2.minimizer)[1]

tspan_test = (0., 5.)
u₀_test = [0.5,0]

prob_test_NN = NeuralODE(model, tspan_test, Tsit5(), p=res2.minimizer)
sol_test_NN = prob_test_NN(u₀_test)

prob_test = remake(prob, tspan = tspan_test, u0=u₀_test)
sol_test = solve(prob_test)

plot(sol_test_NN, vars = (1), label= "NN θ")
plot!(sol_test, vars = (1), label = "truth θ")

plot(sol_test_NN, vars = (2), label= "NN dθ")
plot!(sol_test, vars = (2), label = "truth dθ")
