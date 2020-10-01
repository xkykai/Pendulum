using DifferentialEquations, Plots
using DiffEqFlux, Optim
using Flux
using BlackBoxOptim
using LaTeXStrings
using DelimitedFiles

function undamped_pendulum!(ddθ, dθ, θ, p, t)
    g,L = p
    ddθ[1] = -g/L * sin(θ[1])
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
p_NN = initial_params(model)
NN(du,u,p,t) = model(u,p)
prob_NN = SecondOrderODEProblem{false}(NN, du₀, u₀, tspan_train, p_NN)

function predict(p)
    Array(solve(prob_NN, Tsit5(), p=p, saveat=t_train))
end

function loss_n_ode_vel(p)
    pred = predict(p)
    Flux.mse(dataset_train[1,:], pred[1, :]), pred
end

cb = function (p,l,pred)
    println(l)
    false
end

cb_plot = function (p,l,pred)
    println(l)
    fig = plot(t_train,dataset_train[2,:], ylims=(-6,6), xlims=(0,1.5))
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

res_vel = DiffEqFlux.sciml_train(loss_n_ode_vel, p_NN, Descent(), maxiters=5000)
res_vel = DiffEqFlux.sciml_train(loss_n_ode_vel, res_vel.minimizer, ADAM(0.01), maxiters = 2000)
res_vel = DiffEqFlux.sciml_train(loss_n_ode_vel, res_vel.minimizer, BFGS(), cb = cb_plot_vel)
res_vel = DiffEqFlux.sciml_train(loss_n_ode_vel, res_vel.minimizer, LBFGS(), cb = cb_plot_vel)
res_vel = DiffEqFlux.sciml_train(loss_n_ode_vel, res_vel.minimizer, NADAM(), maxiters = 5000)

plot(t_train,dataset_train[1,:], ylims=(-25,25), xlims=(0,1.5), label = "Training Velocity")
plot!(t_train, predict(res_vel.minimizer)[1,:], label = "Neural Network")
xlabel!("Time")
ylabel!("Angular Velocity")
title!("Trained using only Velocity Data, loss = 0.0147")
savefig("trainvel_undamped_small_vel.pdf")

loss_n_ode_vel(res_vel.minimizer)

writedlm("p_vel_pi_2.csv", res_vel.minimizer)

u₀_test = [1.0]
du₀_test = [5.0]
tspan_test = (0.0,5.0)

prob_test = SecondOrderODEProblem(undamped_pendulum!, du₀_test, u₀_test, tspan_test, p_train)
sol_test = solve(prob_test, Tsit5())

prob_test_NN = SecondOrderODEProblem{false}(NN, du₀_test, u₀_test, tspan_test, p=res_vel.minimizer)
sol_test_NN = solve(prob_test_NN, Tsit5(), p = res_vel.minimizer)


plot(sol_test_NN, vars = (2), label="Neural Network")
plot!(sol_test, vars = (2), label="Test Displacement")
xlabel!("Time")
ylabel!("Angular Displacement")
title!("Velocity Data Training, loss = 0.0147")
savefig("testvel_undamped_small_pos.pdf")
