import numpy as np
import gurobipy as gp
from gurobipy import GRB
env = gp.Env()
env.setParam('OutputFlag', 0)  # in seconds

# Initialization



# h,T
h1 = np.array([0, 0, 500, 100])
h2 = np.array([0, 0, 300, 300])
T = np.array([[-60, 0], [0, -80], [0, 0], [0, 0]])
W = np.array([[6, 10], [8, 5], [1, 0], [0, 1]])
p1, p2 = .4, .6
# Master Problem
c = np.array([100, 150])
a1 = np.array([1, 1])
rhs1 = 120
master = gp.Model('MasterProblem', env=env)
X = master.addMVar((2,), name='X')
master.addConstr(X[0] >= 40, name='X lb')
master.addConstr(X[1] >= 20, name='X lb')
master.addConstr(a1 @ X <= rhs1, name='Const1')  # Ax <= b
master.setObjective(c @ X, GRB.MINIMIZE)  # cx
master.optimize()
x_star = [X[0].x, X[1].x]

# sub-problem 1
s1 = np.array([-24, -28])
sub1 = gp.Model('SubProblem 1', env=env)  # sub-problem
Y1 = sub1.addMVar((4, ), lb=0.0, name='Y1')
sub1.addConstr(np.transpose(W) @ Y1 >= s1, name='sub1 constraint')
sub1.setParam('InfUnbdInfo', 1)


# sub-problem 2
s2 = np.array([-28, -32])
sub2 = gp.Model('SubProblem 2', env=env)  # sub-problem
Y2 = sub2.addMVar((4, ), lb=0.0, name='Y2')
sub2.addConstr(np.transpose(W) @ Y2 >= s2, name='sub2 constraint')
sub2.setParam('InfUnbdInfo', 1)


# Algorithm parameters
w = 0  # w = p*pi*h - p*pi*T@x
termination = float('-inf')  # theta
check_solution = (GRB.UNBOUNDED, GRB.INFEASIBLE, GRB.INF_OR_UNBD)

# Conditional Loop
loop_index = 1

theta = master.addVar(lb=float('-inf'), name='theta')  # add theta variable to the master problem
master.setObjective(c @ X + theta, GRB.MINIMIZE)  # update master's objective function
master.update()
while loop_index <= 10:

    print(f'Iteration {loop_index}', 20*'=')

    sub1.setObjective((h1 - T @ x_star) @ Y1, GRB.MAXIMIZE)
    sub1.update()
    sub1.optimize()

    sub2.setObjective((h2 - T @ x_star) @ Y2, GRB.MAXIMIZE)
    sub2.update()
    sub2.optimize()

    if sub1.status in check_solution or sub2.status in check_solution:
        print('Feasibility cut is needed.')
        if sub1.status == 5:
            master.addConstr((h1 - T @ X) @ Y1.UnbdRay <= 0, name='FeasCut')
            master.update()
        if sub2.status == 5:
            master.addConstr((h2 - T @ X) @ Y2.UnbdRay <= 0, name='FeasCut')
            master.update()
    else:
        print('Both scenarios are bounded, optimality cut must be added')
        print(Y1)
        Pi1 = [Y1[0].x, Y1[1].x, Y1[2].x, Y1[3].x]
        Pi2 = [Y2[0].x, Y2[1].x, Y2[2].x, Y2[3].x]
        print(Pi2, Pi1)
        e = p1 * (Pi1 @ h1) + p2 * (Pi2 @ h2)
        E = p1 * (Pi1 @ T) + p2 * (Pi2 @ T)
        master.addConstr(E @ X + theta >= e, name='OpCut' + str(loop_index))  # add the optimality cut
        master.update()
        master.optimize()
        w = e - E @ x_star
        print(theta)
    master.optimize()
    x_star = [X[0].x, X[1].x]
    loop_index += 1
