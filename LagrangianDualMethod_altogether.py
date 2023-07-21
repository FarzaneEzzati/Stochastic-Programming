import gurobipy as gp
from gurobipy import GRB
import numpy as np
env = gp.Env()
env.setParam('OutputFlag', 0)  # in seconds


if __name__ == '__main__':
    '''
    objective function is the same for all scenarios here. Each row corresponds to a scenario. You can change 
    coefficients for each scenario.
    '''
    var_count = 9
    scen_count = 3

    CQ = np.array([[-150, -230, -260, -238, -210, 170, 150, 36, 10],
                   [-150, -230, -260, -238, -210, 170, 150, 36, 10],
                   [-150, -230, -260, -238, -210, 170, 150, 36, 10]])

    TW = [0, 0, 0]
    TW[0] = np.array([[3, 0,    0, 1, 0, -1,  0,  0,  0],
                      [0, 3.6,  0, 0, 1,  0, -1,  0,  0],
                      [0, 0,  -24, 0, 0,  0,  0,  1, 1]])

    TW[1] = np.array([[2.5, 0,  0,  1, 0, -1,  0,  0,  0],
                      [0,   3,  0,  0, 1,  0, -1,  0,  0],
                      [0,   0, -20, 0, 0,  0,  0,  1, 1]])

    TW[2] = np.array([[2,   0,  0,  1, 0, -1,  0, 0, 0],
                      [0, 2.4,  0,  0, 1,  0, -1, 0, 0],
                      [0,   0, -16, 0, 0,  0,  0, 1, 1]])

    h = np.array([200, 240, 0], int)
    p = np.ones(3)/3

    H1 = np.array([1, 1])
    H2 = np.array([-1, 0])
    H3 = np.array([0, -1])

    lambda0 = np.zeros((2, 3))
    alpha = 0.4
    ZLD = 0
    ZLB = float('-inf')

    '''
    Create the model for Lagrangian problem which is a summation of three sub-problem
    '''
    LGPModels = gp.Model('Lagrangian Problem', env=env)
    V1 = LGPModels.addMVar((9,), lb=0, name=f'vars 1')
    V2 = LGPModels.addMVar((9,), lb=0, name=f'vars 2')
    V3 = LGPModels.addMVar((9,), lb=0, name=f'vars 3')
    LGPModels.addConstr(V1[0] + V1[1] + V1[2] <= 500, name='Axb')
    LGPModels.addConstr(V2[0] + V2[1] + V2[2] <= 500, name='Axb')
    LGPModels.addConstr(V3[0] + V3[1] + V3[2] <= 500, name='Axb')
    LGPModels.addConstr(TW[0] @ V1 >= h, name=f'constraint 1')
    LGPModels.addConstr(TW[1] @ V2 >= h, name=f'constraint 2')
    LGPModels.addConstr(TW[2] @ V3 >= h, name=f'constraint 3')
    LGPModels.addConstr(V1[7] + V1[8] <= 6000, name='lower bound y5 1')
    LGPModels.addConstr(V2[7] + V2[8] <= 6000, name='lower bound y5 2')
    LGPModels.addConstr(V3[7] + V3[8] <= 6000, name='lower bound y5 3')
    LGPModels.update()

    xy_solutions = np.zeros((scen_count, var_count))
    ZLD_FirstTerm = 0
    '''
    Create Lagrangian Dual problem
    '''
    '''a = 10
    LDP = gp.Model('Lagrangian Dual prob.', env=env)
    l1 = LDP.addMVar((3, ), lb=-a, ub=a)
    l2 = LDP.addMVar((3, ), lb=-a, ub=a)
    Zld = LDP.addVar(name='Z for LDP')
    LDP.addConstr(Zld >= 0, name="LDP Cons")
    LDP.setObjective(Zld, GRB.MINIMIZE)
    LDP.update()'''
    '''
    Now the loop: in each iteration, xy_solution, lambda0, ZLD, ZLB are updated
    '''
    v = 0
    while v < 10:  # ZLB < ZLD
        v += 1
        '''
        This part gives us the x and y solutions for D(l), keeping lambda constant.
        XY SOLUTIONS UPDATED
        '''
        LGPModels.setObjective(CQ[0] @ V1 + CQ[1] @ V2 + CQ[2] @ V3 + (H1 @ lambda0) @ V1[0:3] +
                               (H2 @ lambda0) @ V2[0:3] + (H3 @ lambda0) @ V3[0:3], GRB.MAXIMIZE)
        LGPModels.update()
        LGPModels.optimize()

        xy_solutions[0] = V1.x
        xy_solutions[1] = V2.x
        xy_solutions[2] = V3.x

        '''
        LAMBDA UPDATED
        '''
        lambda0 = lambda0 - np.multiply(alpha, [xy_solutions[0][0:3] - xy_solutions[1][0:3], xy_solutions[0][0:3] - xy_solutions[2][0:3]])
        '''
        Check the identicallity of x solutions. 
        ZLB UPDATED
        '''
        if xy_solutions[0].all == xy_solutions[1].all and xy_solutions[0].all == xy_solutions[2].all:
            # Identical
            candidate_ZLB = np.sum([p[i] * (CQ[i]@xy_solutions[i]) for i in range(scen_count)])
            print(candidate_ZLB)
            ZLB = np.max([ZLB, candidate_ZLB])
        else:
            # not identical
            x_avg = xy_solutions.copy()
            for index in range(len(x_avg)):
                avg = np.sum([p[i] * xy_solutions[i][index] for i in range(scen_count)])
                x_avg[0][index] = avg
                x_avg[1][index] = avg
                x_avg[2][index] = avg

            candidate_ZLB = np.sum([p[i] * (CQ[i] @ x_avg[i]) for i in range(scen_count)])
            ZLB = np.max([ZLB, candidate_ZLB])

        print(ZLB, '\n')



