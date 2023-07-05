import gurobipy as gp
from gurobipy import GRB
import numpy as np
env = gp.Env()
env.setParam('OutputFlag', 0)  # in seconds


class CreateModel:
    # initialize the class
    def __init__(self, objective_coefficient, scenario_coefficient, tech_matrix, right_hand_side, prob, g, scenario_name):
        self.c = objective_coefficient
        self.q = scenario_coefficient
        self.rhs = right_hand_side
        self.T = tech_matrix
        self.pi = prob
        self.g = g
        self.s = scenario_name
        self.m = gp.Model(self.s, env=env)
        self.x_count = len(self.c)
        self.y_count = len(self.q)
        # g1 = np.array([1, 1])
        self.X = self.m.addMVar((self.x_count,), vtype=GRB.CONTINUOUS, name='X')
        self.Y = self.m.addMVar((self.y_count,), vtype=GRB.CONTINUOUS, name='Y')
        for i in range(len(self.rhs)):
            self.m.addConstr(self.T[i, 0:self.x_count]@self.X + self.T[i, self.x_count:]@self.Y <= self.rhs[i],
                             name='tech constraints')

    def set_objective(self, dual_l):
        self.m.setObjective(self.pi * (self.c @ self.X + self.q @ self.Y) + dual_l@(self.g @ self.X),
                            sense=GRB.MINIMIZE)
        self.m.update()

    def bounds_x(self, bound, bound_type):

        if bound_type == 'u':
            for j in range(len(bound)):
                self.m.addConstr(self.X[j] <= bound[j], name='upper bound')
            self.m.update()
        elif bound_type == 'l':
            for j in range(len(bound)):
                self.m.addConstr(self.X[j] >= bound[j], name='lower bound')
            self.m.update()
        else:
            raise TypeError('Print specify the type of bound u: Upper and l: Lower')

    def bounds_y(self, bound, bound_type):
        if bound_type == 'u':
            for j in range(len(bound)):
                self.m.addConstr(self.Y[j] <= bound[j], name='upper bound')
        elif bound_type == 'l':
            for j in range(len(bound)):
                self.m.addConstr(self.Y[j] >= bound[j], name='lower bound')
        else:
            raise TypeError('Print specify the type of bound u: Upper and l: Lower')
        self.m.update()

    def optimize(self):
        self.m.optimize()
        if self.m.status in (GRB.INFEASIBLE, GRB.UNBOUNDED, GRB.INF_OR_UNBD):
            status = [0, 0, 0, 'Infeasible', 'Infeasible or Unbounded', 'Unbounded']
            print(f'The problem {self.s} is {status[self.m.status]} ')

    def solution(self):
        if self.m.status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED):
            status = ['Infeasible', 'Infeasible or Unbounded', 'Unbounded']
            print(f'The problem {self.s} is {status[self.m.status-3]} ')
        else:
            print(f'{self.s} solved.')
            value = self.m.ObjVal
            solution = []
            for var in self.m.getVars():
                solution.append(var.x)
            return solution[:self.x_count], solution[self.x_count:],  value


if __name__ == '__main__':
    t = np.array([[1, 1, 0, 0],
                  [-60, 0, 6, 10],
                  [0, -80, 8, 5]])
    c = np.array([100, 150])
    lbx = [40, 20]
    rhs = np.array([120, 0, 0])

    # scenario 1
    q1 = np.array([-24, -28])
    p1 = 0.4
    g1 = np.array([[1, 0], [0, 1]])
    uby = [400, 250]
    m1 = CreateModel(c, q1, t, rhs, p1, g1, 'scenario 1')
    m1.bounds_x(lbx, 'l')
    m1.bounds_y(uby, 'u')

    # scenario 2
    q2 = np.array([-24, -28])
    p2 = .6
    g2 = np.array([[-1, 0], [0, -1]])
    uby = [300, 300]
    m2 = CreateModel(c, q2, t, rhs, p2, g2, 'scenario 2')
    m2.bounds_x(lbx, 'l')
    m2.bounds_y(uby, 'u')

    # MP Model
    mp = gp.Model('Master Problem', env=env)
    v = mp.addVar(vtype=GRB.CONTINUOUS, name='nu')
    l = mp.addMVar((2,), lb=-1000, ub=1000, vtype=GRB.CONTINUOUS, name='lambda')
    mp.setObjective(v, sense=GRB.MAXIMIZE)
    mp_dic = {'nu': 0, 'lambda[0]': 0, 'lambda[1]': 0}

    k = 1
    threshold = 1000
    lambda_con = np.array([1, 1])

    while k <= 10:
        print(f'Iteration {k}', 20*'==', )
        # solve sub-problems
        m1.set_objective(lambda_con)
        m2.set_objective(lambda_con)
        m1.optimize()
        m2.optimize()

        x1, y1, z1 = m1.solution()
        x2, y2, z2 = m2.solution()
        print(x1, x2, y1, y2, z1+z2)
        # generate optimality cut
        if k > 1:
            constraint = mp.getConstrByName('cut')
            mp.remove(constraint)
            mp.update()
        mp.addConstr(v <= z1 + z2 + (g1 @ x1 + g2 @ x2) @ (l - lambda_con), name='cut')
        mp.update()
        mp.optimize()

        if mp.status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED):
            status = ['Infeasible', 'Infeasible or Unbounded', 'Unbounded']
            print(f'The master problem is {status[mp.status-3]} ')
            break
        else:
            for v in mp.getVars():
                mp_dic[v.VarName] = v.X
            print(mp_dic)
            if abs(z1+z2 - mp_dic['nu']) <= threshold:
                print('The algorithms converged.')
                print(mp_dic)
                print(f'Difference: {abs(z1+z2 - mp_dic["nu"])}')
                print(f'scenario 1 X {x1}, y {y1}, Obj {z1}')
                print(f'scenario 2 X {x2}, y {y2}, Obj {z2}')
                break
            else:
                k += 1
                lambda_con[0] = mp_dic['lambda[0]']
                lambda_con[1] = mp_dic['lambda[1]']
