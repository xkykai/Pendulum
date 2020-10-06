using DifferentialEquations, Plots
using DiffEqFlux, Optim
using Flux
using BlackBoxOptim
using LaTeXStrings
using Statistics

function damped_pendulum!(ddθ, dθ, θ, p, t)
    g,L,b = p
    ddθ[1] = -g/L * sin(θ[1]) - b * dθ[1]
end

du₀ = [0.0]
# u₀ = [π-0.1]
u₀ = [0.5]
tspan_train = (0.0, 0.5)
p_train = [9.81, 0.1, 1]

prob = SecondOrderODEProblem(damped_pendulum!, du₀, u₀, tspan_train, p_train)

sol = solve(prob, saveat=0.01)

plot(sol, vars = (2), label = "displacement")
plot(sol, vars = (1), label = "velocity")

dataset_train = Array(sol)
t_train = Array(sol.t)

function feature_scaling(x, mean, std)
    (x .- mean) ./ std
end

dataset_train_dθ = dataset_train[1,:]
mean_train_dθ = mean(dataset_train_dθ)
std_train_dθ = std(dataset_train_dθ)
dataset_train_dθ_reg = feature_scaling(dataset_train_dθ, mean_train_dθ, std_train_dθ)

dataset_train_θ = dataset_train[2,:]
mean_train_θ = mean(dataset_train_θ)
std_train_θ = std(dataset_train_θ)
dataset_train_θ_reg = feature_scaling(dataset_train_θ, mean_train_θ, std_train_θ)

function total_energy(dθ, θ, p)
    g = p[1]
    r = p[2]
    0.5 .* r .^2 .* (dθ .^2) .+ g .* r .* (1 .- cos.(θ))
end

function energy_change(E)
    E[2:end] - E[1:end-1]
end

model = FastChain(FastDense(1, 50, tanh), FastDense(50, 1))
p_NN = initial_params(model)
NN(du,u,p,t) = model(u,p)
prob_NN = SecondOrderODEProblem{false}(NN, du₀, u₀, tspan_train, p_NN)

plot(solve(prob_NN, Tsit5(), p=p_NN, saveat=t_train))

function predict(p)
    Array(solve(prob_NN, Tsit5(), p=p, saveat=t_train))
end

function loss_n_ode(p)
    pred = predict(p)
    E = total_energy(pred[1,:], pred[2,:], p_train)
    ΔE = energy_change(E)
    E_penalty = 0
    for i in 1:length(ΔE)
        if (ΔE[i] > 0)
            E_penalty += ΔE[i] * 100
        end
    end
    loss_dθ_reg = Flux.mse(dataset_train_dθ_reg, feature_scaling(pred[1,:], mean_train_dθ, std_train_dθ))
    loss_θ_reg = Flux.mse(dataset_train_θ_reg, feature_scaling(pred[2,:], mean_train_θ, std_train_θ))
    # sum(abs2, dataset .- pred[1:2, :]), pred
    mean(loss_θ_reg + loss_dθ_reg) + E_penalty, pred
end

cb = function (p,l,pred)
    println(l)
    false
end

cb_plot = function (p,l,pred)
    println(l)
    fig = plot(t_train,dataset_train[2,:], ylims=(-1,1), xlims=(0,1))
    plot!(fig, t_train, pred[2,:])
    display(fig)
    false
end

cb_plot_vel = function (p,l,pred)
    println(l)
    fig = plot(t_train,dataset_train[1,:], ylims=(-25,25), xlims=(0,1.5))
    plot!(fig, t_train, pred[1,:])
    display(fig)
    false
end


println("Starting Gradient Descent")
res = DiffEqFlux.sciml_train(loss_n_ode, p_NN, Descent(), cb = cb, maxiters=5000)
println("Starting ADAM")
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, ADAM(0.01), maxiters = 5000)
println("Starting BFGS")
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, BFGS(), cb = cb_plot)
println("Starting LBFGS")
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, LBFGS(), cb = cb_plot)
# println("Starting NADAM")
# res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, NADAM(), maxiters = 5000)

loss_n_ode(res.minimizer)

prob_extrapolate = SecondOrderODEProblem{false}(NN, du₀, u₀, (0,10),res.minimizer)
plot(solve(prob_NN, Tsit5(), p = res.minimizer, tspan = (0,1)))



plot(solve(prob_NN, Tsit5(), p=res.minimizer), vars=(1), label = "Neural Network")
# plot(solve(prob_extrapolate, Tsit5(), p=res5.minimizer), vars=(2), label = "Neural Network")
plot!(solve(prob, Tsit5(), p=p_train), vars=(1), label = "Training Displacement")


tspan = (0.,1.)
tspan_train = (0., 1.)

sol = solve(prob, saveat=0.01, tspan = tspan)

dataset_train = Array(sol)
t_train = Array(sol.t)

dataset_train_dθ = dataset_train[1,:]
mean_train_dθ = mean(dataset_train_dθ)
std_train_dθ = std(dataset_train_dθ)
dataset_train_dθ_reg = feature_scaling(dataset_train_dθ, mean_train_dθ, std_train_dθ)

dataset_train_θ = dataset_train[2,:]
mean_train_θ = mean(dataset_train_θ)
std_train_θ = std(dataset_train_θ)
dataset_train_θ_reg = feature_scaling(dataset_train_θ, mean_train_θ, std_train_θ)

plot(sol, vars = (2), label = "displacement")
plot!(solve(prob_NN, Tsit5(), p = res.minimizer, tspan = (0,1), saveat=t_train), vars = (2))
plot(sol, vars = (1), label = "velocity")
plot!(solve(prob_NN, Tsit5(), p = res.minimizer, tspan = (0,1)), vars = (1))


prob_NN = SecondOrderODEProblem{false}(NN, du₀, u₀, tspan_train, res.minimizer)

println("Starting Gradient Descent")
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, Descent(), maxiters=5000)
println("Starting ADAM")
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, ADAM(0.01), maxiters = 5000)
println("Starting BFGS")
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, BFGS(), cb = cb_plot)
println("Starting LBFGS")
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, LBFGS(), cb = cb_plot)
println("Starting NADAM")
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, NADAM(), maxiters = 5000)


loss_n_ode(res.minimizer)
