using DifferentialEquations, Plots
using DiffEqFlux, Optim
using Flux
using BlackBoxOptim
using LaTeXStrings

function damped_pendulum!(ddθ, dθ, θ, p, t)
    g,L,b = p
    ddθ[1] = -g/L * sin(θ[1]) - b * dθ[1]
end

du₀ = [0.0]
# u₀ = [π-0.1]
u₀ = [0.5]
tspan_train = (0.0, 1.0)
p_train = [9.81, 0.1, 1]

prob = SecondOrderODEProblem(damped_pendulum!, du₀, u₀, tspan_train, p_train)

sol = solve(prob, saveat=0.01)

plot(sol, vars = (2), label = "displacement")
plot(sol, vars = (1), label = "velocity")

dataset_train = Array(sol)
t_train = Array(sol.t)

dataset_train_dθ = dataset_train[1,:]
dataset_train_θ = dataset_train[2,:]

model = FastChain(FastDense(1, 30, tanh), FastDense(30, 1))
p_NN = initial_params(model)
NN(du,u,p,t) = model(u,p)
prob_NN = SecondOrderODEProblem{false}(NN, du₀, u₀, tspan_train, p_NN)

plot(solve(prob_NN, Tsit5(), p=p_NN, saveat=t_train))

function predict(p)
    Array(solve(prob_NN, Tsit5(), p=p, saveat=t_train))
end

# function predict(p, prob, t)
#     Array(solve(prob, Tsit5(), p=p, saveat=t))
# end

function loss_n_ode(p)
    pred = predict(p)
    Flux.mse(dataset_train[1,:], pred[1, :]), pred
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
println("Starting NADAM")
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, NADAM(), maxiters = 5000)

loss_n_ode(res.minimizer)


prob_extrapolate = SecondOrderODEProblem{false}(NN, du₀, u₀, (0,10),res.minimizer)
plot(solve(prob_NN, Tsit5(), p = res.minimizer, tspan = (0,10)))

plot(solve(prob_NN, Tsit5(), p=res.minimizer), vars=(2), label = "Neural Network")
# plot(solve(prob_extrapolate, Tsit5(), p=res5.minimizer), vars=(2), label = "Neural Network")
plot!(solve(prob, Tsit5(), p=p_train), vars=(2), label = "Training Displacement")
xlabel!("Time")
ylabel!("Angular Displacement")
title!("Velocity Data Training, loss = 0.116")
savefig("trainvel_damped_small_pos.pdf")
