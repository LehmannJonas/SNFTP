# The Service Network Fleet Transition Problem
This repository contains an implementation of the Service Network Fleet Transition Problem (SNFTP) by Lehmann, Gzodjak 
and Winkenbach to inform and guiding the strategic decarbonization of logistics fleets and services. The paper can be
found [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4732766).

## Description
The goal of this repository is to i) provide a python and gurobi implementation of the SNFTP, ii) share the cost data and
assumptions for the analyses presented, and iii) provide plotting functions to recreate the plot styles of the
publication. It should be noted that the _original fleet data of the submission cannot be shared_ due to confidentiality
reasons and the pre-filled configurations use stylized data and are not representative of the actual data.

## Code structure
The code is structured as follows:
- **[data](data)**: Contains publicly available data used in the publication.
    - [cost_data](data/cost_data): Contains a set of csv files with cost structures and parameters as described in the publication.
    - [test_data](data/test_data): Contains a csv files with artificial fleet data and demand data for testing purposes that served the basis for the test configurations.
- **[test_scenario_configs](test_scenario_configs)**: Contains pickle files with the configurations of the analyses presented in the publication based on the test fleet data.
- **[src](src)**: Contains the main objects.
    - [config_class](src/config_class.py): Contains the base class for all optimization model configurations.
    - [model_analyzer_class](src/model_analyzer_class.py): Contains the base class to read out the optimization model results and plot them.
    - [optim_snftp](src/optim_snftp.py): Contains the implementation of the Service Network Fleet Transition Problem.
    - [run_parameters](src/run_parameters.py): Contains a set of parameters used for running the model and analyses.
- **[runMe.py](runMe.py)**: Contains the main script to define and run the optimization model.


## Running the code
To run the code, follow these steps:
1. **Clone the repository:**
```
git clone https://github.com/yourusername/service_network_fleet_transition_problem.git
cd service_network_fleet_transition_problem
```
**2. Install the required packages:** 
- Install the required packages listed in `requirements.txt`.
- It assumes that you have a working license to run Gurobi as solver.


**3. Run the Script:**
- Execute the `runMe.py` script.
```
python runMe.py
```

**4. Additional analyses:**
- To perform additional analyses, change the `scenario_name` variable in the script.
- Update the configurations with your own data as needed.


## Citation
If you use this code in your research, please cite the following publication:

```
@article{Lehmann2024SNFTP,
    title = {{The Service Network Fleet Transition Problem}},
    year = {2024},
    journal = {SSRN},
    author = {Lehmann, Jonas and Gvozdjak, Anne and Winkenbach, Matthias},
    url = {https://ssrn.com/abstract=4732766},
    keywords = {Fleet transition, Integer linear programming, Logistics decarbonization}
}
```

## License
This project is licensed under the MIT [License](LICENSE).