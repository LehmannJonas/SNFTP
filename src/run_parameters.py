"""
Run Parameters
"""

# Parameters for the optimization model
M = 1 # update this if objective function should be scaled
MIPGAP = 0.005 # The MIP gap for the optimization model

# Vehicle classes
NHTSA_CLASSES = ['Class 3', 'Class 6', 'Class 8']

# General parameters
BILLION = 1e9
MILLION = 1e6
THOUSAND = 1e3
PERIOD_ZERO = 2023
LABEL_FONT_SIZE = 14
MONTHS_PER_YEAR = 12

# Plotting parameters
CMU_SERIF_STYLE = True
DEFAULT_MILEAGE_BRACKET = 5000

MAP_VEHICLE_CLASS_TO_COLOR_AND_LINESTYLE = {
    "Class 3":("grey",'dotted'),
    "Class 6":("grey",'dashed'),
    "Class 8":("grey",'dashdot')}

# dict to map vehicle types and colors
MAP_VEHICLE_TYPE_TO_COLOR = {
    'Diesel | Class 3': 'darkgrey',
    'Diesel | Class 6': 'dimgrey',
    'Diesel | Class 8': 'black',
    'CNG | Class 6': 'mediumvioletred',
    'CNG | Class 8': 'darkmagenta',
    'Battery-Electric | Class 3': 'limegreen',
    'Battery-Electric | Class 6': 'forestgreen',
    'Battery-Electric | Class 8': 'darkgreen',
    'Fuel-Cell Electric Hydrogen | Class 6': 'dodgerblue',
    'Fuel-Cell Electric Hydrogen | Class 8': 'royalblue',
}

SCENARIO_NAMES = ["base_case",
                  "sensitivity_25p_BEV_FCEV_vehicle_infra_reduction",
                  "sensitivity_50p_BEV_FCEV_vehicle_infra_reduction",
                  "sensitivity_10p_fossilfuels_increase",
                  "sensitivity_25p_fossilfuels_increase",
                  "sensitivity_50p_fossilfuels_increase",
                  "sensitivity_10p_electricity_hydrogen_reduction",
                  "sensitivity_25p_electricity_hydrogen_reduction",
                  "sensitivity_50p_electricity_hydrogen_reduction",
                  "sensitivity_10p_change_energies_jointly",
                  "sensitivity_25p_change_energies_jointly",
                  "sensitivity_50p_change_energies_jointly",
                  "sensitivity_budget1.5e8",
                  "sensitivity_budget2e8",
                  "sensitivity_25p_demandcharges_reduction",
                  "sensitivity_50p_demandcharges_reduction",
                  "sensitivity_75p_demandcharges_reduction",
                  "sensitivity_1step_decarbonization_goals",
                  "sensitivity_2step_decarbonization_goals",
                  "sensitivity_sigmoid_decarbonization_goals",
                  "sensitivity_linear_decarbonization_goals"]