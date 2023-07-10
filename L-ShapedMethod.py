import numpy as np
import gurobipy as gp
from gurobipy import GRB
env = gp.Env()
env.setParam('OutputFlag', 0)  # in seconds

# Initialization
c = np.array([150, 230, 260], int)
q = np.array([238, 210, -170, -150, -36, -10, 0, 0, 0])
A = np.array([1, 1, 1])
b = 500
ub = 6000
T = np.array([[[3, 0, 0], [0, 3.6, 0], [0, 0, 24]],
              [[2.5, 0, 0], [0, 3, 0], [0, 0, 20]],
              [[2, 0, 0], [0, 2.4, 0], [0, 0, 16]]], int)
W = np.array([[1, 0, -1, 0, 0, 0, -1, 0, 0],
              [0, 1, 0, -1, 0, 0, 0, -1, 0],
              [0, 0, 0, 0, -1, -1, 0, 0, -1]], int)
h = np.array([200, 240, 0], int)
I = np.ones(len(W))
scenario_count = len(T)
prob = np.array([1/3, 1/3, 1/3])
xStar = np.array([0, 0, 0])  # xStar is the solution to the master problem.

# Create models for feasibility cuts
feasibility = []
for index in range(scenario_count):
    feasibility.append(gp.Model(f'scenario {index} - feasibility', env=env))
    Y = feasibility[index].addMVar((9,), name='Y')  # y1, y2, p1, p2, p3, p4, e1, e2, e3
    PV = feasibility[index].addMVar((3,), name='Positive V')
    NV = feasibility[index].addMVar((3,), name='Negative V')
    feasibility[index].addConstr(Y[4] <= ub, name='P3 upper bound')
    feasibility[index].addConstr(W @ Y + PV - NV == h - T[index] @ xStar, name='Feasibility')
    feasibility[index].setObjective(A@PV + A@NV, GRB.MINIMIZE)
    feasibility[index].update()

# Create models for optimality cuts
optimality = []
for index in range(scenario_count):
    optimality.append(gp.Model(f'scenario {index} - optimality', env=env))
    Y = optimality[index].addMVar((9,), name='Y')  # y1, y2, p1, p2, p3, p4, e1, e2, e3
    optimality[index].addConstr(Y[4] <= ub, name='P3 upper bound')
    optimality[index].addConstr(W @ Y >= h - T[index] @ xStar, name='Optimality')
    optimality[index].setObjective(q@Y, GRB.MINIMIZE)
    optimality[index].update()

# Create Master Problem
MP = gp.Model('Master Problem', env=env)
X = MP.addMVar((3,), name='X')
MP.addConstr(A@X <= b, name='AXb')
MP.setObjective(c@X, GRB.MINIMIZE)


# Initialization
w = 0
theta_termination = float('-inf')  # theta
v = 1

while theta_termination < w:
    # Solve MP
    MP.optimize()
    xStar = np.array([X[0].x, X[1].x, X[2].x])
    print(f'MP objective value {MP.ObjVal}')

    if v == 1:
        theta = MP.addVar(lb=float('-inf'), name='theta')
        MP.setObjective(c@X + theta, GRB.MINIMIZE)
        MP.update()
    else:
        theta_termination = theta.x
    # First, update the constraint right hand side of each scenario feasibility constraint. Second, solve the problem\
    # and save the w'
    index = 0
    w_prim = 0
    for m in feasibility:
        constraints = m.getConstrs()[1:]
        m.setAttr('RHS', constraints, h - T[index] @ xStar)
        m.update()
        m.optimize()

        if m.ObjVal > 0:
            # calculate and add feasibility cuts
            duals = m.getAttr('Pi', m.getConstrs()[1:])
            D = np.transpose(duals) @ T[index]
            d = np.transpose(duals) @ h
            MP.addConstr(D @ X >= d, name='Feasibility cut')
            MP.update()

        w_prim += abs(m.ObjVal)
        index += 1

    if w_prim == 0:
        # let's do optimality cuts
        index = 0
        duals = []
        for m in optimality:
            constraints = m.getConstrs()[1:]
            m.setAttr('RHS', constraints, h - T[index] @ xStar)
            m.update()
            m.optimize()
            duals.append(np.transpose(m.getAttr('Pi', m.getConstrs()[1:])))
            index += 1
        E = prob[0]*(duals[0] @ T[0]) + prob[1]*(duals[1] @ T[1]) + prob[2]*(duals[2] @ T[2])
        e = prob[0]*(duals[0] @ h) + prob[1]*(duals[1] @ h) + prob[2]*(duals[2] @ h)
        MP.addConstr(E@X + theta >= e)
        MP.update()

        w = e - E@xStar
        print(f'theta: {theta_termination},     w: {w}')
    else:
        # unfortunately, we need to start over, Duh
        print(f'Iteration {v}, master problem need to be solved again with feasibility cuts')
    v += 1

print(f'You got the optimal solution and it is: ')
for index in range(len(xStar)):
    print(f'X{index+1} = {xStar[index]}')


