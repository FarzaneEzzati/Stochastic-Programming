import gurobipy as gp
from gurobipy import GRB
import numpy as np
env = gp.Env()
env.setParam('OutputFlag', 0)  # in seconds


class CreateModel:
    # initialize the class
    def __init__(self, objective_coefficient, scenario_coefficient, tech_matrix, right_hand_side, scenario_name):
        self.c = objective_coefficient
        self.q = scenario_coefficient
        self.rhs = right_hand_side
        self.T = tech_matrix
        self.s = scenario_name
        self.m = gp.Model(self.s, env=env)
        x_count = len(self.c)
        y_count = len(self.q)
        # g1 = np.array([1, 1])
        self.X = self.m.addMVar(x_count, vtype=GRB.CONTINUOUS, name='X')
        self.Y = self.m.addMVar(y_count, vtype=GRB.CONTINUOUS, name='Y')
        for i in range(len(self.rhs)):
            self.m.addConstr(self.T[i, 0:x_count]@self.X + self.T[i, x_count:]@self.X <= self.rhs[i])
        self.m.setObjective(self.c@self.X + self.q@self.X, sense=GRB.MAXIMIZE)
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
            print(f'The problem is {status[self.m.status]} ')

    def solution(self):
        if self.m.status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED):
            status = [0,0,0,'Infeasible', 'Infeasible or Unbounded', 'Unbounded']
            print(f'The problem is {status[self.m.status]} ')
        else:
            print(f'Problem solve and optimal value {self.m.ObjVal}')
            for v in self.m.getVars():
                print(v.VarName, v.x)


if __name__ == '__main__':
    t1 = np.array([[1, 1, 0, 0],
                   [-60, 0, 6, 10],
                   [-80, 0, 8, 5]])
    c1 = np.array([100, 150])
    q1 = np.array([-24, -28])
    rhs1 = np.array([120, 0, 0])
    lbx = [40, 20]
    uby = [300, 300]
    m1 = CreateModel(c1, q1, t1, rhs1, 'scenario 1')
    m1.bounds_x([90, 40], 'l')
    #m1.bounds_y([500, 100], 'u')
    m1.bounds_y([190, 30], 'l')
    m1.optimize()
    m1.solution()