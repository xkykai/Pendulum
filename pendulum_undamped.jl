using DifferentialEquations, Plots
using DiffEqFlux, Optim
using Flux
using BlackBoxOptim
using LaTeXStrings

function undamped_pendulum!(ddθ, dθ, θ, p, t)
    g,L = p
    ddθ[1] = -g/L * sin(θ[1])
end

function damped_pendulum!(ddθ, dθ, θ, p, t)
    g,L,b = p
    ddθ[1] = -g/L * sin(θ[1]) - b * dθ[1]
end

du₀ = [0.0]
# u₀ = [π-0.1]
u₀ = [π/2]
tspan_train = (0.0, 1.5)
p_train = [9.81, 0.1]

prob = SecondOrderODEProblem(undamped_pendulum!, du₀, u₀, tspan_train, p_train)

sol = solve(prob, saveat=0.01)

plot(sol, vars = (2), label = "displacement")
plot(sol, vars = (1), label = "velocity")

dataset_train = Array(sol)
t_train = Array(sol.t)

dataset_train_dθ = dataset_train[1,:]
dataset_train_θ = dataset_train[2,:]

model = FastChain(FastDense(1, 50, tanh), FastDense(50, 1))
# model = FastChain(FastDense(1, 30, tanh), FastDense(30, 1))
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
    # sum(abs2, dataset .- pred[1:2, :]), pred
    Flux.mse(dataset_train, pred[1:2, :]), pred
end

cb = function (p,l,pred)
    println(l)
    l<0.5
end

cb_plot = function (p,l,pred)
    println(l)
    fig = plot(t_train,dataset_train[2,:], ylims=(-6,6), xlims=(0,1.5))
    plot!(fig, t_train, pred[2,:])
    display(fig)
    l<0.5
end

cb_plot_vel = function (p,l,pred)
    println(l)
    fig = plot(t_train,dataset_train[1,:], ylims=(-25,25), xlims=(0,1.5))
    plot!(fig, t_train, pred[1,:])
    display(fig)
    l<0.5
end
#
# function train_NN(units, du₀, u₀, tspan)
#     model = FastChain(FastDense(1, 50, tanh), FastDense(50, 1))
#     p_NN = initial_params(model)
#     NN(du,u,p,t) = model(u,p)
#     prob_NN = SecondOrderODEProblem{false}(NN, du₀, u₀, tspan_train, p_NN)
#
#     println("Starting Gradient Descent")
#     res = DiffEqFlux.sciml_train(loss_n_ode, p_NN, Descent(), maxiters=500)
#     println("loss = $(Flux.mse(dataset_train, predict(res.minimizer)))")
#     println("Starting ADAM")
#     res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, ADAM(0.01), maxiters = 500)
#     println("loss = $(Flux.mse(dataset_train, predict(res.minimizer)))")
#     println("Starting BFGS")
#     res = DiffEqFlux.sciml_train(loss_n_ode, res2.minimizer, BFGS())
#     println("loss = $(Flux.mse(dataset_train, predict(res.minimizer)))")
#     println("Starting LBFGS")
#     res = DiffEqFlux.sciml_train(loss_n_ode, res3.minimizer, LBFGS())
#     println("loss = $(Flux.mse(dataset_train, predict(res.minimizer)))")
#     println("Starting NADAM")
#     res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, NADAM(), maxiters = 5000)
#     return res
# end
#
# res = train_NN(50, du₀, u₀, tspan_train)

println("Starting Gradient Descent")
res = DiffEqFlux.sciml_train(loss_n_ode, p_NN, Descent(), cb = cb, maxiters=1000)
println("loss = $(Flux.mse(dataset_train, predict(res.minimizer)))")
println("Starting ADAM")
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, ADAM(0.01), maxiters = 500)
println("loss = $(Flux.mse(dataset_train, predict(res.minimizer)))")
println("Starting BFGS")
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, BFGS())
println("loss = $(Flux.mse(dataset_train, predict(res.minimizer)))")
println("Starting LBFGS")
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, LBFGS())
println("loss = $(Flux.mse(dataset_train, predict(res.minimizer)))")
println("Starting NADAM")
res = DiffEqFlux.sciml_train(loss_n_ode, res.minimizer, NADAM(), maxiters = 5000)

prob_extrapolate = SecondOrderODEProblem{false}(NN, du₀, u₀, (0,10),res.minimizer)
plot(solve(prob_NN, Tsit5(), p = res.minimizer, tspan = (0,10)))

plot(solve(prob_NN, Tsit5(), p=res.minimizer), vars=(1), label = "Neural Network")
# plot(solve(prob_extrapolate, Tsit5(), p=res5.minimizer), vars=(2), label = "Neural Network")
plot!(solve(prob, Tsit5(), p=p_train), vars=(1), label = "Truth Data")




# xlabel!("t")
# ylabel!("θ")
# title!("Displacement of Undamped Pendulum")
# savefig("displacement_undamped_2_long.pdf")
#
# plot(solve(prob_extrapolate, Tsit5(), p=res5.minimizer), vars=(1), label = "Neural Network")
# plot!(t,dataset[1,:], label = "Truth Data")
# xlabel!("t")
# ylabel!(L"\dot{\theta}")
# title!("Velocity of Undamped Pendulum")
# savefig("velocity_undamped_2_long.pdf")

# model_vel = FastChain(FastDense(1, 50, tanh), FastDense(50, 1))
# # model = FastChain(FastDense(1, 30, tanh), FastDense(30, 1))
# p_NN_vel = initial_params(model_vel)
# NN_vel(du,p,t) = model_vel(du,p)
# prob_NN_vel = ODEProblem{false}(NN_vel, du₀, tspan_train, p_NN_vel)
#
# function predict_vel(p)
#     Array(solve(prob_NN_vel, Tsit5(), p=p, saveat=t_train))
# end
#
# Array(solve(prob_NN_vel, Tsit5(), p=p_NN_vel, saveat=t_train))



model_vel = FastChain(FastDense(1, 50, tanh), FastDense(50, 1))

initial_params(model_vel)
prob_NN_vel = ODEProblem(model_vel, du₀, tspan_train, p_train)

# prob_NN_vel = NeuralODE(model_vel, tspan_train, Tsit5(), saveat = t_train)

predict_vel(prob_NN_vel.p)

function predict_vel(p)
  Array(prob_NN_vel(du₀, p))
end

function loss_n_ode_vel(p)
    pred = predict_vel(p)
    loss = Flux.mse(dataset_train_dθ, pred)
    return loss, pred
end

du₀ = [1]
res = DiffEqFlux.sciml_train(loss_n_ode_vel, prob_NN_vel.p, Descent(), cb=cb_plot_vel, maxiters=500)
