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
    c = np.array([150, 230, 260, 238, 210, -170, -150, -36, -10])
    CQ = np.array([c, c, c])
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

    H = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 [[-1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                 [[0, 0, 0], [0, 0, 0], [0, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]])

    lambda0 = np.zeros(6)
    alpha = 1
    ZLD = 0
    ZLB = float('-inf')

    '''
    Create the model for Lagrangian problem which is a summation of three sub-problem
    '''
    LGPModels = []
    Vars = []
    for scenario in range(scen_count):
        LGPModels.append(gp.Model(f'Sub Problem {scenario+1}', env=env))
        V = LGPModels[scenario].addMVar((9, ), lb=0, name=f'vars {scenario+1}')
        LGPModels[scenario].addConstr(V[0]+V[1]+V[2] <= 500, name='Axb')
        LGPModels[scenario].addConstr(TW[scenario]@V >= h, name=f'constraint {scenario+1}')
        LGPModels[scenario].addConstr(V[7] + V[8] <= 600, name='lower bound y5')
        LGPModels[scenario].update()
        Vars.append(V)

    xy = np.zeros((scen_count, var_count))
    ZLD_FirstTerm = 0

    v = 0
    while v <= 1:  # ZLB < ZLD
        v += 1
        print(f'Iteration ======== {v}')

        for model in LGPModels:
            scen_ind = LGPModels.index(model)
            SV = Vars[scen_ind]
            model.setObjective(CQ[scen_ind] @ SV + (lambda0 @ H[scen_ind]) @ SV[0:3], GRB.MINIMIZE)
            model.update()
            model.optimize()
            var_val = []
            for var in model.getVars():
                var_val.append(var.x)
            xy[scen_ind] = np.array(var_val)
        lambda0 = alpha*(H[0]@xy[0][0:3] + H[1]@xy[1][0:3] + H[2]@xy[2][0:3])

        if xy[0].all == xy[1].all and xy[0].all == xy[2].all:
            # Identical
            candidate_ZLB = np.sum([p[i] * (CQ[i]@xy[i]) for i in range(scen_count)])
            ZLB = np.max([ZLB, candidate_ZLB])
        else:
            # not identical
            x_avg = xy.copy()
            for index in range(3):
                avg = np.sum([p[i] * xy[i][index] for i in range(scen_count)])
                x_avg[0][index] = avg
                x_avg[1][index] = avg
                x_avg[2][index] = avg

            conditions = []
            for model in LGPModels:
                scen_ind = LGPModels.index(model)
                SV = xy[scen_ind]
                C1 = SV[0] + SV[1] + SV[2] <= 500
                C2 = TW[scen_ind] @ SV >= h
                C3 = SV[7] + SV[8] <= 600
                conditions.append(np.sum(C1)+np.sum(C2)+np.sum(C3))
            if np.sum(conditions) == 15:
                print('Avg is feasible for all')

            candidate_ZLB = np.sum([p[i] * (CQ[i] @ x_avg[i]) for i in range(scen_count)])
            ZLB = np.max([ZLB, candidate_ZLB])


