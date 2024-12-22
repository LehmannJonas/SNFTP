"""
Implementation of the Service Network Fleet Transition Problem formulation
"""

import gurobipy as grb

from src.run_parameters import MIPGAP


class OptimServiceNetworkFleetTransition():
    '''
    Implementation of the Service Network Fleet Transition model
    '''

    # define init function
    def __init__(self, scenario_name='default', config=None):
        self.scenario_name = scenario_name
        self.config = config
        self.define_model()

    def define_constraints(self):
        '''
        Defines the constraints
        '''

        # Define an empty dictionary to store all constraints
        self.constraints = {}

        # Add overall decarbonization targets met constraints
        name = 'overall_decarbonization_targets_met'
        print(f'building constraints: {name}')
        self.constraints[name] = {
            t: self.model.addConstr(
                grb.quicksum(
                    self.w_kjtr[k][j][t][r] * self.config.em_kjtr[k][j][t][r]
                    for k in self.config.K
                    for j in range(self.config.kappa_k[k] + 1)
                    for r in self.config.R if self.config.h_kjtr[k][j][t][r] == 1
                )
                <= (1 - self.config.delta_t[t]) * self.config.E_1,
                name=f'{name}[{t}]'
            )
            for t in self.config.T
        }

        # Add constraints to the dictionary and the model
        name = 'regional_decarbonization_targets_met'
        print(f'building constraints: {name}')
        self.constraints[name] = {
            t: {
                d: self.model.addConstr(
                    grb.quicksum(
                        self.w_kjtr[k][j][t][r] * self.config.em_kjtr[k][j][t][r]
                        for l in self.config.L_d[d]
                        for k in self.config.K
                        for j in range(self.config.kappa_k[k] + 1)
                        for r in self.config.R if (self.config.L_r1[r] == l and self.config.h_kjtr[k][j][t][r] == 1)
                    )
                    <= (1 - self.config.psi_dt[d][t]) * self.config.E_2d[d],
                    name=f'{name}[{t}]'
                )
                for d in self.config.D
            }
            for t in self.config.T
        }


        # Ensure the required number of vehicles are available for each run in each period
        name = 'number_required_vehicles_met_local_runs'
        print(f'building constraints: {name}')
        self.constraints[name] = {
            (r, t): self.model.addConstr(
                grb.quicksum(
                    self.w_kjtr[k][j][t][r]
                    for k in self.config.K
                    for j in range(self.config.kappa_k[k] + 1)
                    if self.config.h_kjtr[k][j][t][r] == 1
                ) == self.config.q_tr[t][r],
                name=f'{name}[{r}][{t}]'
            )
            for r in self.config.R
            for t in self.config.T
        }

        # Ensure available vehicles are assigned to runs
        name = 'assign_available_vehicles_to_runs'
        print(f'building constraints: {name}')
        self.constraints[name] = {
            (k, j, t, l): self.model.addConstr(
                grb.quicksum(self.w_kjtr[k][j][t][r]
                             for r in self.config.R if (self.config.L_r1[r] == l
                                                        and self.config.h_kjtr[k][j][t][r] == 1))
                == self.z_kjtl[k][j][t][l],
                name=f'{name}[{k.replace(" ", "")}][{t}][{j}][{l.replace(" ", "")}]'
            )
            for k in self.config.K
            for j in self.config.J_k[k] if j != self.config.kappa_k[k] + 1
            for t in self.config.T
            for l in self.config.L
        }

        # Set the initial number of vehicles
        name = 'first_set_of_vehicles'
        print(f'building constraints: {name}')
        self.constraints[name] = {(k, l):
                    self.model.addConstr(self.x_ktl[k][1][l] +
                                         self.config.a_kjl[k][0][l] == self.z_kjtl[k][0][1][l],
                                         name=f'{name}[{k.replace(" ", "")}][{l.replace(" ", "")}]')
                                  for k in self.config.K
                                  for l in self.config.L}

        # Set available vehicles for first period
        name = 'set_available_vehicles_first_period'
        print(f'building constraints: {name}')
        self.constraints[name] = {
            (k, j, l): self.model.addConstr(
                self.config.a_kjl[k][j][l] - self.y_kjtl[k][j][1][l] == self.z_kjtl[k][j][1][l],
                name=f'{name}[{k.replace(" ", "")}][{j}][1][{l.replace(" ", "")}]'
            )
            for k in self.config.K
            for j in self.config.J_k[k] if j != 0 and j != self.config.kappa_k[k] + 1
            for l in self.config.L
        }

        # Purchase decisions for the first period
        name = 'purchase_decisions_first_period'
        print(f'building constraints: {name}')
        self.constraints[name] = {}
        for k in self.config.K:
            for t in self.config.T:
                if t != 1:
                    for l in self.config.L:
                        self.constraints[name][(k, t, l)] = \
                            self.model.addConstr(self.x_ktl[k][t][l] == self.z_kjtl[k][0][t][l],
                                                 name=f'{name}[{k.replace(" ", "")}][{t}][{l.replace(" ", "")}]')

        # Setting vehicle size for next period
        name = 'set_vehicle_size_next_period'
        print(f'building constraints: {name}')
        self.constraints[name] = {}
        for k in self.config.K:
            for t in self.config.T:
                if t != 1:
                    for j in self.config.J_k[k]:
                        if (j != 0) and (j != self.config.kappa_k[k] + 1):
                            for l in self.config.L:
                                self.constraints[name][(k, j, t, l)] = self.model.addConstr(
                                    self.z_kjtl[k][j][t][l] == self.z_kjtl[k][j - 1][t - 1][l] -
                                    self.y_kjtl[k][j][t][l],
                                    name=f'{name}[{k.replace(" ", "")}][{j}][{t}][{l.replace(" ", "")}]')

        # Setting vehicles to salvage
        name = 'setting_vehicles_to_salvage'
        print(f'building constraints: {name}')
        self.constraints[name] = {}
        for k in self.config.K:
            for t in self.config.T:
                if t != 1:
                    for l in self.config.L:
                        self.constraints[name][(k, t, l)] = self.model.addConstr(
                            self.y_kjtl[k][self.config.kappa_k[k] + 1][t][l] ==
                            self.z_kjtl[k][self.config.kappa_k[k]][t - 1][l],
                            name=f'{name}[{k.replace(" ", "")}][{t}][{l.replace(" ", "")}]')

        # Salvage existing vehicles
        name = 'salvage_existing_vehicles_first_period'
        print(f'building constraints: {name}')
        self.constraints[name] = {}
        for k in self.config.K:
            for l in self.config.L:
                self.constraints[name][(k, l)] = \
                    self.model.addConstr(self.y_kjtl[k][self.config.kappa_k[k] + 1][1][l]
                                         == self.config.a_kjl[k][self.config.kappa_k[k] + 1][l],
                                         name=f'{name}[{k.replace(" ", "")}][{l.replace(" ", "")}]')

        # Charger composition constraints
        name = 'vehicle_infrastructure_composition_constraints'
        print(f'building constraints: {name}')
        self.constraints[name] = {
            (i, t, l): self.model.addConstr(
                self.config.theta_i[i] *
                grb.quicksum(
                    self.z_kjtl[k][j][t][l] +
                    grb.quicksum(
                        self.w_kjtr[k][j][t][r]
                        for r in self.config.R2 if (self.config.L_r2[r] == l and self.config.h_kjtr[k][j][t][r] == 1)
                    )
                    for k in self.config.K if i in self.config.I_k[k]
                    for j in range(self.config.kappa_k[k] + 1)
                )
                <= self.v_itl[i][t][l],
                name=f'{name}[{i.replace(" ", "")}][{t}][{l.replace(" ", "")}]'
            )
            for i in self.config.I
            for t in self.config.T
            for l in self.config.L
        }


        # Charger watt capacity constraint
        name = 'charger_watt_capacity_constraint'
        print(f'building constraints: {name}')
        self.constraints[name] = {}
        for t in self.config.T:
            for l in self.config.L:
                self.constraints[name][(t, l)] = \
                    self.model.addConstr(grb.quicksum(self.config.g_i[i] *
                                                      self.v_itl[i][t][l]
                                                      for i in self.config.I_E)
                                         <= self.config.G_tl[t][l],
                                         name=f'{name}[{t}][{l.replace(" ", "")}]')

        # Charger space capacity constraint
        name = 'charger_space_capacity_constraint'
        print(f'building constraints: {name}')
        self.constraints[name] = {}
        for t in self.config.T:
            for l in self.config.L:
                self.constraints[name][(t, l)] = \
                    self.model.addConstr(grb.quicksum(self.v_itl[i][t][l] for i in self.config.I)
                                         <= self.config.H_l[l], name=f'{name}[{t}][{l.replace(" ", "")}]')

        # Setting initial vehicle fueling and charging infrastructure numbers
        name = 'setting_initial_fueling_infrastructure_numbers'
        print(f'building constraints: {name}')
        self.constraints[name] = {}
        for i in self.config.I:
            for l in self.config.L:
                self.constraints[name][(i, l)] = \
                    self.model.addConstr(self.v_itl[i][1][l] ==
                                         self.config.e_il[i][l] +
                                         self.u_itl[i][1][l] * self.config.d_i[i],
                                         name=f'{name}[{i.replace(" ", "")}][{l.replace(" ", "")}]')

        # Setting vehicle fueling and charging infrastructure numbers
        name = 'setting_vehicle_fueling_infrastructure_numbers'
        print(f'building constraints: {name}')
        self.constraints[name] = {}
        for i in self.config.I:
            for t in self.config.T:
                if t != 1:
                    for l in self.config.L:
                        self.constraints[name][(i, t, l)] = \
                            self.model.addConstr(self.v_itl[i][t][l] ==
                                                 self.v_itl[i][t - 1][l] +
                                                 self.u_itl[i][t][l] * self.config.d_i[i],
                                                 name=f'{name}[{i.replace(" ", "")}][{t}][{l.replace(" ", "")}]')

        # Setting the site preparation constraint for installing fueling and charging infrastructure
        name = 'setting_site_preparation_constraint_for_infrastructure'
        print(f'building constraints: {name}')
        self.constraints[name] = {}
        for i in self.config.I:
            for t in self.config.T:
                for l in self.config.L:
                    self.constraints[name][(i, t, l)] = \
                        self.model.addConstr(self.v_itl[i][t][l] <=
                                             self.epsilon_itl[i][t][l] * self.config.H_l[l],
                                             name=f'{name}[{i.replace(" ", "")}][{t}][{l.replace(" ", "")}]')

        # Setting the initial value for epsilon if infrastructure does not already exist
        name = 'setting_initial_value_for_epsilon1'
        print(f'building constraints: {name}')
        self.constraints[name] = {}
        for i in self.config.I:
            for l in self.config.L:
                if self.config.e_il[i][l] == 0:
                    self.constraints[name][(i, 1, l)] = \
                        self.model.addConstr(self.epsilon_itl[i][1][l] == self.xi_itl[i][1][l],
                                             name=f'{name}[{i.replace(" ", "")}][1][{l.replace(" ", "")}]')

        # Setting the initial value for epsilon if infrastructure already exists
        name = 'setting_initial_value_for_epsilon2'
        print(f'building constraints: {name}')
        self.constraints[name] = {}
        for i in self.config.I:
            for l in self.config.L:
                if self.config.e_il[i][l] > 0:
                    self.constraints[name][(i, 1, l)] = \
                        self.model.addConstr(self.epsilon_itl[i][1][l] == 1,
                                             name=f'{name}[{i.replace(" ", "")}][1][{l.replace(" ", "")}]')

        # Setting the initial value for xi if infrastructure already exists
        name = 'setting_initial_value_for_xi'
        print(f'building constraints: {name}')
        self.constraints[name] = {}
        for i in self.config.I:
            for l in self.config.L:
                if self.config.e_il[i][l] > 0:
                    for t in self.config.T:
                        self.constraints[name][(i, t, l)] = \
                            self.model.addConstr(self.xi_itl[i][t][l] == 0,
                                                 name=f'{name}[{i.replace(" ", "")}][{t}][{l.replace(" ", "")}]')

        # Set the flow conservation for epsilon
        name = 'setting_flow_conservation_for_epsilon'
        print(f'building constraints: {name}')
        self.constraints[name] = {}
        for i in self.config.I:
            for t in self.config.T:
                if t != 1:
                    for l in self.config.L:
                        self.constraints[name][(i, t, l)] = \
                            self.model.addConstr(self.epsilon_itl[i][t][l] ==
                                                 self.epsilon_itl[i][t - 1][l] +
                                                 self.xi_itl[i][t][l],
                                                 name=f'{name}[{i.replace(" ", "")}][{t}][{l.replace(" ", "")}]')

        # Ensure that a site can only be prepared once per technology over the time horizon
        # by setting the sum of xi's over the time horizon to 1 per site and technology
        name = 'setting_sum_of_xi_over_time_horizon_to_1'
        print(f'building constraints: {name}')
        self.constraints[name] = {}
        for i in self.config.I:
            for l in self.config.L:
                self.constraints[name][(i, l)] = \
                    self.model.addConstr(grb.quicksum(self.xi_itl[i][t][l] for t in self.config.T) <= 1,
                                         name=f'{name}[{i.replace(" ", "")}][{l.replace(" ", "")}]')

        # Budget constraint
        name = 'budget_constraint'
        print(f'building constraints: {name}')
        self.constraints[name] = {}
        for t in self.config.T:
            self.constraints[name][t] = \
                self.model.addConstr(grb.quicksum(
                    grb.quicksum(self.x_ktl[k][t][l] * self.config.cp_ktl[k][t][l] -
                                 grb.quicksum(self.config.s_kjtl[k][j][t][l] *
                                              self.y_kjtl[k][j][t][l]
                                              for j in range(1, self.config.kappa_k[k] + 2))
                                 for k in self.config.K)
                    + grb.quicksum(self.u_itl[i][t][l] * self.config.ci_itl[i][t][l] * self.config.d_i[i]
                                   for i in self.config.I)
                    for l in self.config.L) / self.config.M
                                     <= self.config.b_t[t],
                                     name=f'{name}[{t}]')

        # Maximum average fleet age at the end
        name = 'max_average_fleet_age_at_end_by_class'
        print(f'building constraints: {name}')
        self.constraints[name] = {}
        for t in [self.config.t_F]:
            self.constraints[name][t] = {}
            for l in self.config.L:
                self.constraints[name][t][l] = {}
                for c in self.config.C:
                    self.constraints[name][t][l][c] = self.model.addConstr(
                        grb.quicksum(self.z_kjtl[k][j][t][l] * (self.config.Gamma_c[c] - j)
                                     for k in self.config.K_c[c]
                                     for j in range(self.config.kappa_k[k] + 1))
                        >= 0,
                        name=f'{name}[{t}][{l.replace(" ", "")}]')

    def define_decision_variables(self):
        '''
        Defines the decision variables
        '''

        print('Defining decision variables')

        # Indicator variable if location l has been prepared for installing fueling and charging
        # infrastructure i at the start of period t
        print('Defining epsilon_itl')
        self.epsilon_itl = {
            i: {
                t: {
                    l: self.model.addVar(vtype=grb.GRB.BINARY,
                                         name=f'epsilon[{i.replace(" ", "")}][{t}][{l.replace(" ", "")}]')
                    for l in self.config.L
                }
                for t in self.config.T
            }
            for i in self.config.I
        }

        # Indicator variable if location l is being prepared for installing fueling and charging
        # infrastructure i at the beginning of period t
        print('Defining xi_itl')
        self.xi_itl = {
            i: {
                t: {
                    l: self.model.addVar(vtype=grb.GRB.BINARY,
                                         name=f'xi[{i.replace(" ", "")}][{t}][{l.replace(" ", "")}]')
                    for l in self.config.L
                }
                for t in self.config.T
            }
            for i in self.config.I
        }

        # Number of vehicle fueling and charging infrastructure points i purchased and
        # installed at the start of period t at location l
        print('Defining u_itl')
        self.u_itl = {
            i: {
                t: {
                    l: self.model.addVar(vtype=grb.GRB.INTEGER,
                                         name=f'u[{i.replace(" ", "")}][{t}][{l.replace(" ", "")}]')
                    for l in self.config.L
                }
                for t in self.config.T
            }
            for i in self.config.I
        }


        # Number of available vehicle fueling and charging infrastructure i during period t at location l
        print('Defining v_itl')
        self.v_itl = {
            i: {
                t: {
                    l: self.model.addVar(vtype=grb.GRB.INTEGER,
                                         name=f'v[{i.replace(" ", "")}][{t}][{l.replace(" ", "")}]')
                    for l in self.config.L
                }
                for t in self.config.T
            }
            for i in self.config.I
        }

        # Number of vehicles of type k and age j assigned to run type r in R in period t
        # if the parameter h_kjtr is 1
        print('Defining w_kjtr')
        self.w_kjtr = {
            k: {
                j: {
                    t: {
                        r: self.model.addVar(vtype=grb.GRB.INTEGER, name=f'w_[{k.replace(" ", "")}][{j}][{t}][{r}]')
                        for r in self.config.R if self.config.h_kjtr[k][j][t][r] == 1
                    }
                    for t in self.config.T
                }
                for j in self.config.J_k[k] if j != self.config.kappa_k[k] + 1
            }
            for k in self.config.K
        }

        # Number of vehicles of type k purchased at the beginning of period t at location l
        print('Defining x_ktl')
        self.x_ktl = {
            k: {
                t: {
                    l: self.model.addVar(vtype=grb.GRB.INTEGER,
                                         name=f'x[{k.replace(" ", "")}][{t}][{l.replace(" ", "")}]')
                    for l in self.config.L
                }
                for t in self.config.T
            }
            for k in self.config.K
        }

        # Number of vehicles of type k and age j salvaged at the beginning of period t at location l
        print('Defining y_kjtl')
        self.y_kjtl = {
            k: {
                j: {
                    t: {
                        l: self.model.addVar(vtype=grb.GRB.INTEGER,
                                             name=f'y[{k.replace(" ", "")}][{j}][{t}][{l.replace(" ", "")}]')
                        for l in self.config.L
                    }
                    for t in self.config.T
                }
                for j in self.config.J_k[k] if j != 0
            }
            for k in self.config.K
        }


        # Number of available vehicles of type k and age j in period t at location l
        print('Defining z_kjtl')
        self.z_kjtl = {
            k: {
                j: {
                    t: {
                        l: self.model.addVar(vtype=grb.GRB.INTEGER,
                                             name=f'z[{k.replace(" ", "")}][{j}][{t}][{l.replace(" ", "")}]')
                        for l in self.config.L
                    }
                    for t in self.config.T
                }
                for j in self.config.J_k[k] if j != self.config.kappa_k[k] + 1
            }
            for k in self.config.K
        }

    def define_model(self):
        '''
        Defines the model
        '''
        print('Defining optimization model')

        # initialization
        self.model = grb.Model(name='ServiceNetworkFleetTransition')

        # define variables
        self.define_decision_variables()

        # define objective function
        self.define_objective_function()

        # define constraints
        self.define_constraints()

        # set the MIP gap
        self.model.setParam('MIPGap', MIPGAP)

        # tune the model
        self.model.setParam('Heuristics', 0.8)


    def define_objective_function(self):
        '''
        Defines the objective function
        '''

        print('Define objective term 1')
        term1 = grb.quicksum(self.config.beta ** (t - 1) *
                             grb.quicksum(grb.quicksum(self.config.cp_ktl[k][t][l] * self.x_ktl[k][t][l]
                                                       -
                                                       grb.quicksum(
                                                           self.config.s_kjtl[k][j][t][l] * self.y_kjtl[k][j][t][l]
                                                           for j in range(1, self.config.kappa_k[k] + 1 + 1))
                                                       +
                                                       self.config.cm_ktl[k][t][l] *
                                                       self.z_kjtl[k][self.config.alpha_k[k]][t][l]
                                                       for l in self.config.L) +
                                          grb.quicksum(self.config.co_kjtr[k][j][t][r] *
                                                       self.w_kjtr[k][j][t][r]
                                                       for j in range(self.config.kappa_k[k] + 1)
                                                       for r in self.config.R if self.config.h_kjtr[k][j][t][r] == 1)
                                          for k in self.config.K) for t in self.config.T)

        # charging infrastructure investments, and demand charges
        print('Define objective term 2')
        term2 = grb.quicksum(
            self.config.beta ** (t - 1) *
            grb.quicksum(
                grb.quicksum(
                    self.config.ci_itl[i][t][l] *
                    self.u_itl[i][t][l] * self.config.d_i[i]
                    +
                    self.config.cs_itl[i][t][l] *
                    self.xi_itl[i][t][l]
                    for i in self.config.I)
                +
                self.config.cd_tl[t][l] * grb.quicksum(
                    self.config.theta_i[i] *
                    self.config.g_i[i] *
                    grb.quicksum(
                        self.z_kjtl[k][j][t][l] +
                        grb.quicksum(self.w_kjtr[k][j][t][r]
                                     for r in self.config.R2
                                     if (l in self.config.L_r2[r] and self.config.h_kjtr[k][j][t][r] == 1)
                                     )
                        for k in self.config.K
                        if i in self.config.I_k[k]
                        for j in range(self.config.kappa_k[k] + 1))
                    for i in self.config.I_E)
                for l in self.config.L)
            for t in self.config.T)

        ## end-of-horizon costs ##

        # end-of-horizon operating costs
        print('Define objective term 3')
        term3 = grb.quicksum(self.config.beta ** (self.config.t_F + i - 1) *
                             grb.quicksum(self.config.co_kjtr[k][j + i][self.config.t_F][r]
                                          * self.w_kjtr[k][j][self.config.t_F][r]
                                          for r in self.config.R if self.config.h_kjtr[k][j][self.config.t_F][r] == 1)
                             for k in self.config.K
                             for j in range(self.config.kappa_k[k])
                             for i in range(1, self.config.kappa_k[k] - j + 1))

        # end-of-horizon salvage revenues
        print('Define objective term 4')
        term4 = grb.quicksum(self.config.beta ** (self.config.t_F + self.config.kappa_k[k] - j)
                             * self.config.s_kjtl[k][self.config.kappa_k[k] + 1][self.config.t_F][l] *
                             self.z_kjtl[k][j][self.config.t_F][l]
                             for k in self.config.K
                             for l in self.config.L
                             for j in range(self.config.kappa_k[k] + 1))

        # end-of-horizon midlife costs
        print('Define objective term 5')
        term5 = grb.quicksum(
            self.config.beta ** (self.config.t_F + self.config.alpha_k[k] - j - 1) *
            self.config.cm_ktl[k][self.config.t_F][l] * self.z_kjtl[k][j][self.config.t_F][l]
            for k in self.config.K
            for l in self.config.L
            for j in range(self.config.alpha_k[k]))

        self.model.setObjective((term1 + term2 + term3 - term4 + term5) / self.config.M, grb.GRB.MINIMIZE)