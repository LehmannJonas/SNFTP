"""
Class for storing the model configurations, i.e. sets and parameters
"""

import pickle


class ConfigBase(object):
    """
    Config class for the optimization models sets and parameters
    """

    def __init__(self, scenario_name):
        self.scenario_name = scenario_name
        self.load_config_from_file(scenario_name)

    def load_config_from_file(self, scenario_name):
        try:
            with open(f'test_scenario_configs/{scenario_name}_config.pkl', 'rb') as file:
                config = pickle.load(file)
                self.__dict__.update(config.__dict__)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file for scenario '{scenario_name}' not found.") from e
