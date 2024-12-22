"""
Script to run the service network fleet transition model
"""

import time

import gurobipy as grb
from matplotlib import rcParams
import matplotlib.font_manager as font_manager

import src.config_class as config_class
import src.model_analyzer_class as ma
import src.optim_snftp as optim_snftp
import src.run_parameters as rp


# import and set matplotlib font default to latex font
if rp.CMU_SERIF_STYLE:
    font_dir = ['_cmu_serif/']
    for font in font_manager.findSystemFonts(font_dir):
        font_manager.fontManager.addfont(font)
    rcParams['font.family'] = 'CMU Serif'


def run_model(scenario_name, config=None):
    """
    Runs the optimization model for scenario `scenario_name`
    """

    # track the runtime
    tick = time.time()

    # define the optimization model
    optim = optim_snftp.OptimServiceNetworkFleetTransition(scenario_name=scenario_name, config=config)

    # run the model
    optim.model.optimize()

    # get the status
    status = optim.model.getAttr('status')
    print(f"{optim.model.ModelName} model status: {status}")

    # print the runtime
    tock = time.time()
    print(f"Runtime: {tock - tick} seconds")

    # test if model returns infeasible
    if status == grb.GRB.INFEASIBLE:
        # if not, write the IIS
        optim.model.computeIIS()

        optim.model.write(f"IIS.ILP")

        # relax and solve again
        vars = optim.model.getVars()
        ubpen = [1.0] * optim.model.numVars
        optim.model.feasRelax(1, False, vars, None, ubpen, None, None)
        optim.model.optimize()
        raise Exception(f'{optim.model.ModelName} model infeasible')

    return optim, status



if __name__ == '__main__':

    # set the scenario
    scenario_name = "base_case"
    print(f'Running Scenario {scenario_name}')

    # set up the configuration
    config = config_class.ConfigBase(scenario_name)

    # create and run the model
    optim, status = run_model(scenario_name, config)

    # define the model analyzer object
    model_analyser = ma.ModelAnalyzer(optim, config)

    # plot the results
    model_analyser.plot_model_results()
