"""
Class to save and analyse the results of the optimization model
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from src.run_parameters import *


class ModelAnalyzer:

    def __init__(self, optim, config):
        self.optim_model = optim
        self.config = config
        self.create_results_dict()

    def create_results_dict(self):
        """
        Creates a dictionary to store results in terms of the decision variables
        """

        # define the results dictionary
        self.results_dict = {}

        self.results_dict['scenario_name'] = self.optim_model.scenario_name
        self.results_dict['objective_value'] = self.optim_model.model.objVal
        self.results_dict['runtime'] = self.optim_model.model.Runtime

        # get the fleet composition
        self.results_dict['fleet_composition'] = self.get_fleet_composition()

        # get the vehicle purchase decisions
        self.results_dict['vehicle_purchase_decisions'] = self.get_purchase_decisions()

        # get the vehicle salvage decisions
        self.results_dict['vehicle_salvage_decisions'] = self.get_salvage_decisions()

        # get the infrastructure purchase decisions
        self.results_dict['infrastructure_purchase_decisions'] = self.get_infrastructure_purchase_decisions()

        # get the site preparation decisions
        self.results_dict['site_preparation_decisions'] = self.get_site_preparation_decisions()

        # get the infrastructure compositions
        self.results_dict['infrastructure_composition'] = self.get_infrastructure_composition()

        # get the run assignments
        self.results_dict['run_assignments'] = self.get_run_assignments()

        # demand charges
        self.results_dict['demand_charges'] = self.get_demand_charges()

        # cost for setting up and installing vehicle fueling and charging infrastructure
        self.results_dict['infrastructure_costs'] = self.get_infrastructure_costs()

        # midlife costs
        self.results_dict['midlife_costs'] = self.get_midlife_costs()

        # vehicle purchase costs
        self.results_dict['vehicle_purchase_costs'] = self.get_vehicle_purchase_cost_dict()

        # operating costs
        self.results_dict['operating_costs'] = self.get_operating_costs_dict()

        # salvage revenues
        self.results_dict['salvage_revenues'] = self.get_salvage_revenues_dict()

    def get_demand_charges(self):
        """
        Get the demand charges
        :return: dictionary with the demand charges by time and location
        """
        # get the demand charges
        demand_charges_dict = {}
        for t in self.config.T:
            demand_charges_dict[t] = {}
            for l in self.config.L:
                demand_charges_dict[t][l] = (self.config.beta ** (t - 1) * self.config.cd_tl[t][l] *
                                             sum([self.config.theta_i[i] * self.config.g_i[i] *
                                                  sum([self.optim_model.z_kjtl[k][j][t][l].x +
                                                       sum([self.optim_model.w_kjtr[k][j][t][r].x
                                                            for r in self.config.R2
                                                            if l in self.config.L_r2[r]
                                                            if self.optim_model.config.h_kjtr[k][j][t][r] == 1])
                                                       for k in self.config.K if
                                                       i in self.config.I_k[k]
                                                       for j in range(self.config.kappa_k[k] + 1)])
                                                  for i in self.config.I_E])
                                             )

        return demand_charges_dict

    def get_fleet_composition(self):
        '''
        Retrieve the fleet composition from the optimization model
        :return: a dictionary with the fleet composition by technology and time
        '''

        # set up dict to store fleet composition
        fleet_composition_dict = {k: {j: {t: {l: self.optim_model.z_kjtl[k][j][t][l].x
                                              for l in self.config.L}
                                          for t in self.config.T}
                                      for j in self.config.J_k[k] if j != self.config.kappa_k[k] + 1}
                                  for k in self.config.K}

        return fleet_composition_dict

    def get_infrastructure_composition(self):
        """
        Get the infrastructure composition
        """

        # get the composition of the infrastructure
        infrastructure_composition_dict = {
            i:
                {t:
                     {l:
                          self.optim_model.v_itl[i][t][l].x
                      for l in self.config.L}
                 for t in self.config.T}
            for i in self.config.I}

        return infrastructure_composition_dict

    def get_infrastructure_costs(self):
        """
        Get the infrastructure costs
        :return: dict with infrastructure costs by time and location
        """

        # get the infrastructure costs
        infrastructure_costs_dict = {}
        for t in self.optim_model.config.T:
            infrastructure_costs_dict[t] = {}
            for l in self.optim_model.config.L:
                infrastructure_costs_dict[t][l] = self.optim_model.config.beta ** (t - 1) * \
                                                  sum([self.optim_model.config.ci_itl[i][t][l] *
                                                       self.results_dict['infrastructure_purchase_decisions'][i][t][l] *
                                                       self.optim_model.config.d_i[i] +
                                                       self.optim_model.config.cs_itl[i][t][l] *
                                                       self.results_dict['site_preparation_decisions'][i][t][l]
                                                       for i in self.optim_model.config.I])

        return infrastructure_costs_dict

    def get_infrastructure_purchase_decisions(self):
        """
        Get the infrastructure purchase decisions
        :return: a dictionary with the infrastructure purchase decisions by technology, time and location
        """
        # we aggregate the infrastructure purchase decisions by the infrastructure types in I
        # via the decision variable u for each time period
        infrastructure_purchase_dict = {
            i: {
                t: {
                    l: self.optim_model.u_itl[i][t][l].x
                    for l in self.config.L
                }
                for t in self.config.T
            }
            for i in self.config.I
        }

        return infrastructure_purchase_dict

    def get_midlife_costs(self):
        """
        get all midlife costs
        :return: a dictionary with the midlife costs by time and location
        """

        # get the midlife costs
        midlife_costs_dict = {}
        for t in self.optim_model.config.T:
            midlife_costs_dict[t] = {}
            for l in self.optim_model.config.L:
                midlife_costs_dict[t][l] = self.optim_model.config.beta ** (t - 1) * \
                                           sum([self.optim_model.config.cm_ktl[k][t][l] *
                                                self.results_dict['fleet_composition'][k][
                                                    self.optim_model.config.alpha_k[k]][t][l]
                                                for k in self.optim_model.config.K])
                if t == self.optim_model.config.t_F:
                    midlife_costs_dict[t][l] += sum(
                        [self.optim_model.config.beta ** (self.optim_model.config.t_F +
                                                          self.optim_model.config.alpha_k[k] - j - 1) *
                         self.optim_model.config.cm_ktl[k][t][l] *
                         self.results_dict['fleet_composition'][k][j][t][l]
                         for k in self.optim_model.config.K
                         for j in range(self.optim_model.config.alpha_k[k])])

        return midlife_costs_dict

    def get_operating_costs_dict(self):
        """
        get all operating costs
        :return: a dictionary with the operating costs by time and run
        """

        # get the operating costs
        operating_costs_dict = {}
        # loop over all local runs
        for t in self.optim_model.config.T:
            operating_costs_dict[t] = {}
            for r in self.optim_model.config.R:
                operating_costs_dict[t][r] = self.optim_model.config.beta ** (t - 1) * \
                                             sum([self.optim_model.config.co_kjtr[k][j][t][r] *
                                                  self.results_dict['run_assignments'][k][j][t][r]
                                                  for k in self.optim_model.config.K
                                                  for j in range(self.optim_model.config.kappa_k[k] + 1)
                                                  if self.optim_model.config.h_kjtr[k][j][t][r] == 1])
                if t == self.optim_model.config.t_F:
                    operating_costs_dict[t][r] += sum(
                        [self.optim_model.config.beta ** (self.optim_model.config.t_F + i - 1) *
                         self.optim_model.config.co_kjtr[k][j + i][t][r] *
                         self.results_dict['run_assignments'][k][j][t][r]
                         for k in self.optim_model.config.K
                         for j in range(self.optim_model.config.kappa_k[k])
                         for i in range(1, self.optim_model.config.kappa_k[k] - j + 1)
                         if self.optim_model.config.h_kjtr[k][j][t][r] == 1])

        return operating_costs_dict

    def get_purchase_decisions(self):
        '''
        Retrieve the purchase decisions
        :return: a dictionary with the purchase decisions by technology, time and location
        '''

        # set up dict to store technologies
        technology_purchase_dict = {
            k:
                {t:
                     {l:
                          self.optim_model.x_ktl[k][t][l].x
                      for l in self.config.L}
                 for t in self.config.T}
            for k in self.config.K}

        return technology_purchase_dict

    def get_run_assignments(self):
        """
        Gets the run assignments
        :return: a dictionary with the run assignments
        """

        w_kjtr = {k:
                      {j:
                           {t:
                                {r: self.optim_model.w_kjtr[k][j][t][r].x
                                 for r in self.config.R if self.config.h_kjtr[k][j][t][r] == 1}
                            for t in self.config.T}
                       for j in self.config.J_k[k] if j != self.config.kappa_k[k] + 1}
                  for k in self.config.K}

        return w_kjtr

    def get_salvage_decisions(self):
        """
        Get the salvage decisions
        :return: a dictionary with the salvage decisions by technology, time and location
        """

        salvage_dict = {
            k: {
                j: {
                    t: {
                        l: self.optim_model.y_kjtl[k][j][t][l].x
                        for l in self.config.L
                    }
                    for t in self.config.T
                }
                for j in self.config.J_k[k] if j != 0
            }
            for k in self.config.K
        }

        return salvage_dict

    def get_salvage_revenues_dict(self):
        """
        get all salvage revenues
        :return: a dictionary with the salvage revenues by time and location
        """

        # get the salvage revenues
        salvage_revenues_dict = {}
        for t in self.optim_model.config.T:
            salvage_revenues_dict[t] = {}
            for l in self.optim_model.config.L:
                salvage_revenues_dict[t][l] = self.optim_model.config.beta ** (t - 1) * \
                                              sum([self.optim_model.config.s_kjtl[k][j][t][l] *
                                                   self.results_dict['vehicle_salvage_decisions'][k][j][t][l]
                                                   for k in self.optim_model.config.K
                                                   for j in range(1, self.optim_model.config.kappa_k[k] + 1 + 1)])
                if t == self.optim_model.config.t_F:
                    salvage_revenues_dict[t][l] += sum(
                        [self.optim_model.config.beta ** (
                                    self.optim_model.config.t_F + self.optim_model.config.kappa_k[k] - j) *
                         self.optim_model.config.s_kjtl[k][self.optim_model.config.kappa_k[k] + 1][
                             self.optim_model.config.t_F][
                             l] *
                         self.results_dict['fleet_composition'][k][j][self.optim_model.config.t_F][l]
                         for k in self.optim_model.config.K
                         for j in range(1, self.optim_model.config.kappa_k[k] + 1)])

        return salvage_revenues_dict

    def get_site_preparation_decisions(self):
        """
        Gets the site preparation decisions
        """

        # set up dict to store the site preparation decisions
        site_preparation_dict = {
            i:
                {t:
                     {l:
                          self.optim_model.xi_itl[i][t][l].x
                      for l in self.config.L}
                 for t in self.config.T}
            for i in self.config.I
        }

        return site_preparation_dict


    def get_vehicle_purchase_cost_dict(self):
        """
        Gets the vehicle purchase costs
        :return: a dictionary with the purchase costs by time and location
        """

        # get the purchase costs
        purchase_costs_dict = {}
        for t in self.optim_model.config.T:
            purchase_costs_dict[t] = {}
            for l in self.optim_model.config.L:
                purchase_costs_dict[t][l] = self.optim_model.config.beta ** (t - 1) * \
                                            sum([self.optim_model.config.cp_ktl[k][t][l] *
                                                 self.results_dict['vehicle_purchase_decisions'][k][t][l]
                                                 for k in self.optim_model.config.K])

        return purchase_costs_dict


    def plot_cost_components_aggregated(self, aggregated_by='cost_object'):
        """
        plot cost components aggregated
        """

        if aggregated_by == 'cost_object':
            # get the costs dictionary
            cost_dicts = dict()
            cost_dicts['vehicle purchase costs'] = sum([sum(dic.values())
                                                        for key, dic
                                                        in self.results_dict['vehicle_purchase_costs'].items()])
            cost_dicts['demand charges'] = sum([sum(dic.values())
                                                for key, dic
                                                in self.results_dict['demand_charges'].items()])
            cost_dicts['operating costs'] = sum([sum(dic.values())
                                                 for key, dic
                                                 in self.results_dict['operating_costs'].items()])
            cost_dicts['salvage revenues'] = -sum([sum(dic.values())
                                                   for key, dic
                                                   in self.results_dict['salvage_revenues'].items()])
            cost_dicts['vehicle midlife costs'] = sum([sum(dic.values())
                                                       for key, dic
                                                       in self.results_dict['midlife_costs'].items()])
            cost_dicts['vehicle infrastructure costs'] = sum([sum(dic.values())
                                                              for key, dic
                                                              in self.results_dict['infrastructure_costs'].items()])

            # Extract keys and values
            keys = list(cost_dicts.keys())
            original_values = list(cost_dicts.values())
            sorted_values = original_values.copy()
            sorted_values.sort(reverse=True)
            sorted_keys = [keys[original_values.index(v)] for v in sorted_values]

            fig, ax = plt.subplots(figsize=(5, 4))
            bottom, x_label = 0, ('Total cost',)

            # Get the colors from the colormap
            colormap = plt.cm.get_cmap('Greys')
            color_range = np.linspace(0.1, 0.9, len(sorted_values))

            # Assign a color to each category
            colors = [colormap(i) for i in color_range]

            for cost_label, cost_item in zip(sorted_keys, sorted_values):
                cur_cost = cost_item / BILLION
                if cost_item != 0:
                    ax.bar(x_label, cur_cost, 0.3, label=cost_label.capitalize(),
                           bottom=bottom, color=colors[sorted_values.index(cost_item)])
                    bottom += cur_cost

            # add a legend
            plt.legend(bbox_to_anchor=(1.04, 0), loc='lower left', frameon=False, fontsize=LABEL_FONT_SIZE)

            fig.suptitle('Overall costs by components', fontsize=LABEL_FONT_SIZE)
            fig.subplots_adjust(top=0.85)

            # set the size of the ticks
            plt.tick_params(axis='both', which='major', labelsize=LABEL_FONT_SIZE)

            plt.ylabel('USD (Billions)', fontsize=LABEL_FONT_SIZE)
            plt.ylim(0)

            # adjust layout
            plt.tight_layout()

            # show the plot
            plt.show()

            print(f'Total costs: {np.around(self.results_dict["objective_value"])}')


        elif aggregated_by == 'time':
            cost_dicts = dict()
            # loop over the locations to add the costs

            for t in self.config.T:
                cost_dicts[t] = 0  # initialize
                cost_dicts[t] += sum(
                    [self.results_dict['vehicle_purchase_costs'][t][l]
                     for l in self.results_dict['vehicle_purchase_costs'][t].keys()])
                cost_dicts[t] += sum(
                    [self.results_dict['demand_charges'][t][l] for l in self.config.L])
                cost_dicts[t] += sum(
                    [self.results_dict['operating_costs'][t][r]
                     for r in self.results_dict['operating_costs'][t].keys()])
                cost_dicts[t] += -sum(
                    [self.results_dict['salvage_revenues'][t][l]
                     for l in self.results_dict['salvage_revenues'][t].keys()])
                cost_dicts[t] += sum(
                    [self.results_dict['midlife_costs'][t][l]
                     for l in self.results_dict['midlife_costs'][t].keys()])
                cost_dicts[t] += sum([self.results_dict['infrastructure_costs'][t][l] for l in
                                      self.results_dict['infrastructure_costs'][t].keys()])

            # Extract keys and values
            keys = list(cost_dicts.keys())
            years = [period + PERIOD_ZERO for period in keys]
            values = list(cost_dicts.values())
            costs = [val / BILLION for val in values]

            # Create the bar plot
            plt.figure(figsize=(6, 4))
            plt.bar(years, costs, color='darkgray')

            # set the size of the ticks
            plt.tick_params(axis='both', which='major', labelsize=LABEL_FONT_SIZE)

            # Labels
            plt.xlabel('Year', fontsize=LABEL_FONT_SIZE)
            plt.ylabel('USD (Billions)', fontsize=LABEL_FONT_SIZE)
            plt.xlim(min(years) - 1, max(years) + 2)
            plt.xticks(np.arange(min(years), max(years) + 2, 5))

            # Show the plot
            plt.tight_layout()

            # show
            plt.show()

        else:
            raise NotImplementedError

    def plot_decarbonization_curves(self):
        """
        plot decarbonization curves
        """

        emissions_dict = dict()
        for t in self.config.T:
            emissions_dict[t] = dict()
            for r in self.config.R:
                emissions_dict[t][r] = sum(
                    [self.config.em_kjtr[k][j][t][r] *
                     self.results_dict['run_assignments'][k][j][t][r]
                     for k in self.config.K
                     for j in range(self.config.kappa_k[k] + 1)
                     if self.config.h_kjtr[k][j][t][r] == 1])

        # get the emissions per year from the operations
        emissions_by_year = [sum([emissions_dict[t][r] for r in self.config.R])
                             for t in emissions_dict.keys()]
        print(f"Total emissions over time: {np.around(sum(emissions_by_year),2)}")


        # get the emissions per year and vehicle class from the operations
        emissions_by_year_and_class = dict()
        vehicle_classes = NHTSA_CLASSES

        for vc in vehicle_classes:
            emissions_by_year_and_class[vc] = dict()
            for t in emissions_dict.keys():
                emissions_by_year_and_class[vc][t] = sum([self.config.em_kjtr[k][j][t][r]
                                                          * self.results_dict['run_assignments'][k][j][t][r]
                                                          for k in self.config.K if
                                                          self.config.K[k]['Class'] == vc
                                                          for j in range(self.config.kappa_k[k] + 1)
                                                          for r in self.config.R
                                                          if self.config.h_kjtr[k][j][t][r] == 1])

        years = list(emissions_dict.keys())
        emissions_reductions = [self.config.E_1 *
                                (1 - self.config.delta_t[t])
                                for t in years]

        # copy the steps to provide the stepwise plotting
        for vc in emissions_by_year_and_class:
            emissions_by_year_and_class[vc] = [item / MILLION for item in
                                               list(emissions_by_year_and_class[vc].values()) for _ in range(2)]
        years = [PERIOD_ZERO + item + 0.999 * index for item in years for index in range(2)]
        plt.figure(figsize=(9, 5))

        # plot the target emission lines
        emissions_reductions = [item / MILLION for item in emissions_reductions for _ in range(2)]
        plt.plot(years[:-1], emissions_reductions[:-1],
                 label='Emissions reduction targets', color='black')

        emissions_by_year = [item / MILLION for item in emissions_by_year for _ in range(2)]
        # plot the emissions
        plt.plot(years[:-1], emissions_by_year[:-1], label='Total emissions', color='dimgrey')
        for vc in vehicle_classes:
            plt.plot(years[:-1],
                     emissions_by_year_and_class[vc][:-1],
                     color=MAP_VEHICLE_CLASS_TO_COLOR_AND_LINESTYLE[vc][0],
                     linestyle=MAP_VEHICLE_CLASS_TO_COLOR_AND_LINESTYLE[vc][1],
                     label=f'{vc} emissions')

        plt.legend(frameon=False, fontsize=LABEL_FONT_SIZE)
        plt.tick_params(axis='x', labelsize=LABEL_FONT_SIZE)  # For x-axis tick labels
        plt.tick_params(axis='y', labelsize=LABEL_FONT_SIZE)  # For y-axis tick labels
        plt.title(f'Decarbonization curve  \n Model:' + self.results_dict["scenario_name"])
        plt.text(0.5, 0.04, 'Year', ha='center', fontsize=LABEL_FONT_SIZE)
        plt.xlabel('Year', fontsize=LABEL_FONT_SIZE)
        plt.xlim((min(years), max(years[:-1]) + 1))
        plt.xticks(np.arange(min(years), max(years[:-1]) + 2, 4))
        plt.ylabel('Emissions [MtCO2e]', fontsize=LABEL_FONT_SIZE)
        plt.ylim(0)
        plt.tight_layout()
        plt.show()

    def get_total_emissions(self, divided_by=MILLION):
        """
        Gets the total emissions over all timesteps by iterating over the periods and run assignments to retrieve emissions
        :return: total_emissions
        """

        total_emissions = sum(
            [self.config.em_kjtr[k][j][t][r] *
             self.results_dict['run_assignments'][k][j][t][r]
             for t in self.config.T
             for r in self.config.R
             for k in self.config.K
             for j in range(self.config.kappa_k[k] + 1)
             if self.config.h_kjtr[k][j][t][r] == 1])

        return np.around(total_emissions / divided_by, 3)

    def plot_fleet_composition(self, aggregated_by=None):
        '''
        Plots the transition of the fleet composition by technology and time
        '''

        # get the fleet composition
        fleet_composition_dict = self.results_dict['fleet_composition']

        if aggregated_by == 'vehicle class':
            yaxis = 'Share of vehicle technology in fleet'

            fleet_composition_dict_aggregated = {}
            vehicle_classes = NHTSA_CLASSES

            for vehicle_class in vehicle_classes:
                fleet_composition_dict_aggregated[vehicle_class] = {}
                for k in self.config.K:
                    if self.config.K[k]['Class'] == vehicle_class:
                        fleet_composition_dict_aggregated[vehicle_class][k] = list()
                        for t in self.config.T:
                            fleet_composition_dict_aggregated[vehicle_class][k].append(
                                sum([fleet_composition_dict[k][j][t][l]
                                     for l in self.config.L
                                     for j in self.config.J_k[k]
                                     if j != self.config.kappa_k[k] + 1
                                     ]))

            n_plots = len(vehicle_classes)
            fig, axs = plt.subplots(1, n_plots, figsize=(4 * n_plots, 6), sharex=False, sharey=False)

            # Iterate over the dictionary and create the subplots
            for i, (vehicle_class, data) in enumerate(fleet_composition_dict_aggregated.items()):
                techs = []
                for tech_key, values in sorted(data.items()):
                    x_values = [yr + PERIOD_ZERO for yr in range(1, len(values) + 1)]
                    if all(v == 0 for v in values):
                        axs[i].plot(x_values, np.array(values) /
                                    self.config.MAX_FLEET_SIZE, label=f"{tech_key}",
                                    color=MAP_VEHICLE_TYPE_TO_COLOR[tech_key], alpha=0)
                    else:
                        axs[i].plot(x_values, np.array(values) /
                                    self.config.MAX_FLEET_SIZE, label=f"{tech_key}",
                                    color=MAP_VEHICLE_TYPE_TO_COLOR[tech_key])
                    axs[i].set_title(f"{vehicle_class}", fontsize=LABEL_FONT_SIZE + 1)
                    axs[i].set_xlabel('Year', fontsize=LABEL_FONT_SIZE)
                    axs[i].set_xlim(min(x_values), max(x_values) + 1)
                    axs[i].set_xticks(np.arange(min(x_values), max(x_values) + 2, 4))
                    axs[i].set_ylim(0, 0.57)
                    axs[i].set_yticks(np.arange(0, 0.51, 0.1))
                    axs[i].tick_params(axis='x', labelsize=LABEL_FONT_SIZE)
                    axs[i].tick_params(axis='y', labelsize=LABEL_FONT_SIZE)
                    techs.append(tech_key)

                custom_lines = [Line2D([0], [0], color=MAP_VEHICLE_TYPE_TO_COLOR[tech_key], lw=2) for tech_key in
                                techs]
                abbr_techs = [cur_tech.split(' | ')[0] for cur_tech in techs]
                axs[i].legend(custom_lines, abbr_techs, bbox_to_anchor=(0.5, -0.2), loc='upper center',
                              frameon=False, fontsize=LABEL_FONT_SIZE)

            fig.axes[0].set_ylabel(yaxis, rotation=90, fontsize=LABEL_FONT_SIZE)
            title_cost_val = str(round(int(self.results_dict['objective_value']) / BILLION, 3))
            title_emissions_val = str(self.get_total_emissions())
            title_runtime_val = str(int(round(int(self.results_dict['runtime']) / 60, 0)))
            title_str = 'Cost: $' + title_cost_val + 'bn' + ' | Emissions: ' + title_emissions_val + ' MtCO2e | Runtime: ' + title_runtime_val + ' min'
            fig.suptitle(t=title_str, ha='center', fontsize=LABEL_FONT_SIZE + 2)
            fig.tight_layout()
            fig.subplots_adjust(left=0.05, top=0.85)  # Adjust suptitle position

        elif aggregated_by == 'location':
            location_keys = list(self.config.L)

            # print the fleet composition
            title = 'Fleet composition by vehicle type, location and year | Cost: ' + \
                    str(np.round(self.results_dict['objective_value']))
            yaxis = 'Share of vehicle technology in fleet'

            # aggregate it by location
            fleet_composition_dict_aggregated_by_location = dict()
            for l in location_keys:
                fleet_composition_dict_aggregated_by_location[l] = dict()
                for k in self.config.K:
                    fleet_composition_dict_aggregated_by_location[l][k] = list()
                    for t in self.config.T:
                        fleet_composition_dict_aggregated_by_location[l][k].append(
                            sum([fleet_composition_dict[k][j][t][l]
                                 for j in self.results_dict['fleet_composition'][k].keys()]))

            # Number of subplots
            n_plots = len(location_keys)

            if n_plots == 1:
                plt.figure(figsize=(10, 5))
                for k in self.config.K:
                    y_values = np.array(fleet_composition_dict_aggregated_by_location[
                                            list(fleet_composition_dict_aggregated_by_location.keys())[0]][k]) \
                               / self.config.MAX_FLEET_SIZE
                    x_values = range(1, len(y_values) + 1)
                    plt.plot(x_values, y_values,
                             label=f'{k}',
                             color=MAP_VEHICLE_TYPE_TO_COLOR[k])
                plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False)
                plt.title(title)
                plt.ylabel(yaxis)
                plt.xlabel('Period')
                plt.xlim(1)
                plt.ylim(0, 1)

            else:
                # Create figure and axis objects
                fig, axs = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=True, sharey=True)

                # Set overall title
                fig.suptitle(title, fontsize=16)

                # Iterate over the dictionary and create the subplots
                for i, (location_index, data) in enumerate(fleet_composition_dict_aggregated_by_location.items()):
                    for tech_key, values in data.items():
                        x_values = range(1, len(values) + 1)
                        axs[i].plot(x_values, np.array(values) /
                                    self.config.L_dict[location_index]['fleet_size'],
                                    label=f"{tech_key}",
                                    color=MAP_VEHICLE_TYPE_TO_COLOR[tech_key])
                        axs[i].set_title(f"Location: {location_index}")
                        axs[i].set_xlabel('Period')
                        axs[i].set_ylabel(yaxis)
                        axs[i].set_xlim(1, len(values) + 1)
                        if i == 0:
                            axs[i].legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False)

                    # Adjust layout for better aesthetics
                    fig.tight_layout()
                    fig.subplots_adjust(top=0.90)  # Adjust suptitle position

        elif aggregated_by == 'vehicle class':
            # print the fleet composition
            title = 'Fleet composition by vehicle class and year | Cost: ' + \
                    str(np.round(self.results_dict['objective_value']))
            yaxis = 'Share of vehicle technology in fleet'
            fleet_composition_dict_aggregated = {}
            vehicle_classes = NHTSA_CLASSES

            for vehicle_class in vehicle_classes:
                fleet_composition_dict_aggregated[vehicle_class] = {}
                # sort the keys
                k_sorted = sorted(self.config.K.keys())
                for k in k_sorted:
                    if self.config.K[k]['Class'] == vehicle_class:
                        fleet_composition_dict_aggregated[vehicle_class][k] = list()
                        for t in self.config.T:
                            fleet_composition_dict_aggregated[vehicle_class][k].append(
                                sum([fleet_composition_dict[k][j][t]
                                     for j in self.config.J_k[k]
                                     if j != self.config.kappa_k[k] + 1]))

            n_plots = len(vehicle_classes)
            # Create figure and axis objects
            fig, axs = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=True, sharey=True)

            # Set overall title
            fig.suptitle(title, fontsize=16)

            # Iterate over the dictionary and create the subplots
            for i, (vehicle_class, data) in enumerate(fleet_composition_dict_aggregated.items()):
                for tech_key, values in data.items():
                    x_values = range(1, len(values) + 1)
                    axs[i].plot(x_values, np.array(values) / self.config.MAX_FLEET_SIZE,
                                label=f"{tech_key}",
                                color=MAP_VEHICLE_TYPE_TO_COLOR[tech_key])
                    axs[i].set_title(f"{vehicle_class}")
                    axs[i].set_xlabel('Period')
                    axs[i].set_ylabel(yaxis)
                    axs[i].set_xlim(1, len(values) + 1)
                    axs[i].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

            # Adjust layout for better aesthetics
            fig.tight_layout()
            fig.subplots_adjust(top=0.90)  # Adjust suptitle position

        plt.tight_layout()
        plt.show()

    def plot_mileage_histograms_by_technology(self):
        """
        Plots the histograms of mileage driven by technology
        params: t: time period
        """

        # we define the number of rows by the number of classes
        n_rows = len(NHTSA_CLASSES)

        # we define the number of columns by the number of periods we want to show
        n_cols = 4
        periods = [1, 5, 11, 17]

        # Create figure and axis objects
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=False, sharey=False)

        # we loop over for each vehicle class
        vehicle_classes = NHTSA_CLASSES

        # we loop over the rows and columns
        for row_index, vehicle_class in enumerate(vehicle_classes):
            # sort the technology list
            technology_list_sorted = sorted([k for k in self.config.K if
                                             self.config.K[k]['Class'] == vehicle_class])

            for col_index, period in enumerate(periods):
                # get a dictionary to set the mileage per technology
                miles_dict = {}
                for k in technology_list_sorted:
                    miles_dict[k] = [[self.config.R_dict[r]['mileage_bracket']] *
                                     int(self.results_dict['run_assignments'][k][j][period][r])
                                     for j in self.config.J_k[k]
                                     if j != self.config.kappa_k[k] + 1
                                     for r in self.config.R
                                     if self.config.R_dict[r]['nhtsa_class'] == vehicle_class
                                     if self.results_dict['run_assignments'][k][j][period][r] > 0]
                    miles_dict[k] = [item for sublist in miles_dict[k] for item in sublist]
                    # drop all zeros from the list
                    miles_dict[k] = [x / THOUSAND for x in miles_dict[k] if x != 0]
                miles_list_sorted = [miles_dict[k] for k in technology_list_sorted]

                # get the number of unique entries for the number of bins
                n_bins = int(max([item for sublist in miles_list_sorted for item in sublist]) / (
                            DEFAULT_MILEAGE_BRACKET / THOUSAND))

                # update the labels by taking out the specific class
                technology_list_sorted_labels = [tech.split(' | ')[0] for tech in technology_list_sorted]

                # plot stocked histogram for ax
                axs[row_index, col_index].hist(miles_list_sorted, stacked=True,
                                               color=[MAP_VEHICLE_TYPE_TO_COLOR[k] for k in technology_list_sorted],
                                               bins=n_bins, alpha=0.8)

                # include legend only from rightmost subplots
                if col_index == len(periods) - 1:
                    custom_lines = [Line2D([0], [0], color=MAP_VEHICLE_TYPE_TO_COLOR[tech_key], lw=2) for tech_key
                                    in technology_list_sorted]
                    axs[row_index, col_index].legend(custom_lines, technology_list_sorted_labels,
                                                     bbox_to_anchor=(1.2, 0.5), loc='center left',
                                                     frameon=False, fontsize=LABEL_FONT_SIZE)

        # set single x/y axis labels across all subplots
        # currently uses raw numbers for position rather than with respect to figsize so a little bit tacky
        yaxis = 'Number of vehicles'
        fig.text(0.5, 0.04, 'Annual average mileage in thousands of miles', ha='center',
                 fontsize=LABEL_FONT_SIZE + 1)
        fig.text(0.06, 0.5, yaxis, va='center', rotation='vertical', fontsize=LABEL_FONT_SIZE + 1)

        # set the font size for all x and y ticks
        for ax in axs.flat:
            ax.tick_params(axis='x', labelsize=LABEL_FONT_SIZE)
            ax.tick_params(axis='y', labelsize=LABEL_FONT_SIZE)

        # restrict the xlim to 70 for the first row, 85 for the second row,
        # and 300 for third row axis
        for ax in axs[0]:
            ax.set_xlim(0, 70)
            ax.set_xticks(np.arange(0, 70, 20))
        for ax in axs[1]:
            ax.set_xlim(0, 85)
            ax.set_xticks(np.arange(0, 85, 25))
        for ax in axs[2]:
            ax.set_xlim(0, 299)
            ax.set_xticks(np.arange(0, 299, 100))

        # set single row/column labels for each row/column
        col_labels = [f'{PERIOD_ZERO + period}' for period in periods]
        for ax, col in zip(axs[0], col_labels):
            ax.set_title(col, fontsize=LABEL_FONT_SIZE)

        for ax, row in zip(axs[:, 0], vehicle_classes):
            ax.set_ylabel(row, rotation='vertical', fontsize=LABEL_FONT_SIZE)

        fig.tight_layout()
        fig.subplots_adjust(left=0.12, bottom=0.1, top=0.9)  # Adjust suptitle position

        plt.show()



    def plot_infrastructure_purchase_decisions(self):
        """
        plots the infrastructure purchase decisions
        """

        # we aggregate the infrastructure purchase decisions by the infrastructure types in I
        # via the decision variable u for each time period
        infrastructure_purchase_dict_aggregated = {
            i: {t: sum(l_dict.values()) for t, l_dict in t_dict.items()}
            for i, t_dict in self.results_dict['infrastructure_purchase_decisions'].items()}

        # plot the purchase decisions
        title = f'Infrastructure purchase decisions \n by infrastructure type and year'
        plt.figure(figsize=(10, 5))
        infrastructures = list(self.config.I)
        infrastructures.sort()
        for i in infrastructures:
            y_values = list(infrastructure_purchase_dict_aggregated[i].values())
            x_values = range(1, len(y_values) + 1)
            plt.plot(x_values, y_values,
                     label=f'{i}')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.title(title)
        yaxis = 'Number of units'
        plt.ylabel(yaxis)
        plt.xlabel('Period')
        plt.xlim(1)
        plt.tight_layout()
        plt.show()

    
    def plot_salvage_decisions(self):
        """
        Plot the salvage decisions based on the variable y
        """

        # we aggregate the salvage decisions by location
        salvage_dict_aggregated = {}
        for k in self.config.K:
            salvage_dict_aggregated[k] = {}
            for t in self.config.T:
                salvage_dict_aggregated[k][t] = sum([self.results_dict['vehicle_salvage_decisions'][k][j][t][l]
                                                     for j in self.config.J_k[k] if j != 0
                                                     for l in self.config.L])

        title = 'Salvage decisions by vehicle type and year'
        yaxis = 'Number of vehicles'
        plt.figure(figsize=(10, 5))
        tech_list = list(self.config.K.keys())
        tech_list.sort()

        for k in tech_list:
            y_values = list(salvage_dict_aggregated[k].values())
            x_values = range(1, len(y_values) + 1)
            plt.plot(x_values, y_values,
                     label=f'{k}',
                     color=MAP_VEHICLE_TYPE_TO_COLOR[k])
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.title(title)
        plt.ylabel(yaxis)
        plt.xlabel('Period')
        plt.xlim(1)
        # plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.show()


    def plot_vehicle_purchase_decisions(self):
        '''
        Plots the vehicle purchase decisions by technology and time
        '''

        # get the purchase decisions
        fleet_purchase_dict = self.results_dict['vehicle_purchase_decisions']

        fleet_purchase_dict_aggregated = {}
        for k in self.config.K:
            fleet_purchase_dict_aggregated[k] = {}
            for t in self.config.T:
                fleet_purchase_dict_aggregated[k][t] = sum([fleet_purchase_dict[k][t][l]
                                                            for l in self.config.L])

        title = 'Vehicle purchase decisions by vehicle type and year'
        yaxis = 'Number of vehicles'
        plt.figure(figsize=(10, 5))
        techs = list(self.config.K.keys())
        techs.sort()
        for k in techs:
            y_values = np.array(list(fleet_purchase_dict_aggregated[k].values()))
            x_values = range(1, len(y_values) + 1)
            plt.plot(x_values, y_values,
                     label=f'{k}',
                     color=MAP_VEHICLE_TYPE_TO_COLOR[k])
        # plt.legend()
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.title(title)
        plt.ylabel(yaxis)
        # plt.ylim(0, optim_model.config.MAX_FLEET_SIZE)
        plt.xlim(1)
        plt.xlabel('Period')
        plt.tight_layout()
        plt.show()


    def plot_model_results(self):
        """
        Plot the results of the optimization model
        """

        # plot the fleet composition
        self.plot_fleet_composition(aggregated_by='vehicle class')
        self.plot_fleet_composition(aggregated_by='location')

        # plot the purchase decisions
        self.plot_vehicle_purchase_decisions()
        self.plot_salvage_decisions()
        self.plot_infrastructure_purchase_decisions()
        self.plot_cost_components_aggregated()
        self.plot_cost_components_aggregated(aggregated_by='time')

        # plot decarbonization curves
        self.plot_decarbonization_curves()

        # plot the histograms of mileage driven by technology
        self.plot_mileage_histograms_by_technology()
