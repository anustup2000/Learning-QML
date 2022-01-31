#!/usr/bin/env python
# coding: utf-8

# In[1]:


""" 
MIT License
Copyright (c) 2022 Maxime Dion <maxime.dion@usherbrooke.ca>
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np

from qiskit.opflow import ExpectationFactory, StateFn, CircuitStateFn, CircuitSampler
from qiskit import Aer, execute

from IPython.display import display, clear_output


def embed_data(parametrized_circuit,data_params,data_xs):
    """Replace the parameters of a parametrized QuantumCircuit with data.
    Produce a list of QuantumCircuit, one per data point.
    Args:
        parametrized_circuit (QuantumCircuit): The data embedding circuit
        data_params (ParameterVector or list[Parameter]): Parameters where to input the data
        data_xs (np.array): Data points.
    Returns:
        List of QuantumCircuit: One QuantumCircuit per data point.
    """
    
    data_circuits = list()
    for data_x in data_xs:
        data_value_dict = {p:v for (p,v) in zip(data_params, data_x)}
        data_circuit = parametrized_circuit.bind_parameters(data_value_dict)
        data_circuits.append(data_circuit)
    
    return data_circuits


def prepare_all_circuits(model_circuit,data_params,model_params,data_xs,model_values,add_measurements=False):
    """Replace the model parameters of a parametrized QuantumCircuit with parameter values.
    Then replace the parameters of a parametrized QuantumCircuit with data.
    Produce a list of QuantumCircuit, one per data point.
    Args:
        model_circuit ([type]): The model circuit
        data_params ([type]): Parameters where to input the data
        model_params ([type]): Parameters where to input the model parameter values
        data_xs ([type]): Data points.
        model_values ([type]): Model parameter values
        add_measurements (bool, optional): Add a measurement at the end of the circuit. Defaults to False.
    Returns:
        List of QuantumCircuit: One QuantumCircuit per data point.
    """

    model_value_dict = {p:v for (p,v) in zip(model_params, model_values)}
    classifier_circuit = model_circuit.bind_parameters(model_value_dict)
    if add_measurements:
        classifier_circuit.measure_all()
        
    all_circuits = embed_data(classifier_circuit,data_params,data_xs)
    
    return all_circuits

def all_results_to_expectation_values(all_results):
    """Convert results from running a list 1 qubit QuantumCircuit into Z expectation values.
    Select between statevector and counts method based on the backend used
    Args:
        all_results : Results from runnin all the circuit
    Returns:
        np.array: All the Z expectation values.
    """

    if all_results.backend_name == 'statevector_simulator':

        return all_statevectors_to_expectation_values(all_results)

    else:

        return all_counts_to_expectation_values(all_results.get_counts())


def all_counts_to_expectation_values(all_counts):
    """Convert a list of 1 qubit QuantumCircuit counts into Z expectation values.
    Results from the qasm_simulator or an actual backend.
    Args:
        all_counts (list of dict): The counts resulting of running all the QuantumCircuit. One per data point.
    Returns:
        np.array: All the Z expectation values.
    """

    n_data = len(all_counts)
    expectation_values = np.zeros((n_data,))
    eigenvalues = {'0': 1, '1': -1}
    for i, counts in enumerate(all_counts):
        tmp1 = 0
        tmp2 = 0
        for key, value in counts.items():
            tmp1 += value * eigenvalues[key]
            tmp2 += value
        expectation_values[i] = tmp1/tmp2

    return expectation_values

def all_statevectors_to_expectation_values(all_results):
    """Convert the statevectors resulting of the simulation of a list of 1 qubit QuantumCircuit into Z expectation values.
    Results from the statevector_simulator.
    Args:
        all_counts (list of dict): The result of running all the QuantumCircuit.
    Returns:
        np.array: All the Z expectation values.
    """

    n_circuits = len(all_results.results)
    all_statevectors = np.zeros((n_circuits,2),dtype = complex)
    for i in range(n_circuits):
        all_statevectors[i,:] = all_results.get_statevector(i)

    pauli_z_eig = np.array([1.,-1.])
    expectation_values = np.real(np.einsum('ik,ik,k->i',all_statevectors,np.conjugate(all_statevectors),pauli_z_eig))

    return expectation_values


def eval_cost_fct_linear(expectation_values,target_values):
    """Convert expectation values into cost using a linear distance.
    Args:
        expectation_values (np.array): Values between -1 and 1.
        target_values (np.array): Values -1 or 1
    Returns:
        [np.array]: The computed cost of each data point.
    """

    product_zt = expectation_values*target_values
    all_costs = (1 - product_zt)/2
    return all_costs

def eval_cost_fct_quadratic(expectation_values,target_values):
    """Convert expectation values into cost using a quadratic distance.
    Args:
        expectation_values (np.array): Values between -1 and 1.
        target_values (np.array): Values -1 or 1
    Returns:
        [np.array]: The computed cost of each data point.
    """
    product_zt = expectation_values*target_values
    all_costs = ((1 - product_zt)/2)**2
    return all_costs


def spsa_optimizer_callback(nb_fct_eval, params, fct_value, stepsize, step_accepted, train_history):

    train_history.append((nb_fct_eval,params,fct_value))
    clear_output(wait=True)
    display(f'evaluations : {nb_fct_eval} loss: {fct_value:0.4f}')


def train_classifier(optimizer,eval_cost_fct,quantum_instance,model_circuit,data_params,model_params,data_xs,data_ys,initial_point):
    """Train a classification model quantum circuit.
    Args:
        optimizer (Qiskit Optimizer): The optimizer used to minimize the cost function
        eval_cost_fct (function): Computes the cost of data points given expectation values and target values
        quantum_instance (Qiskit QuantumInstance): On which to run the QuantumCircuits.
        model_circuit (QuantumCircuit): The parametrized QuantumCircuit model.
        data_params ([type]): Parameters where to input the data
        model_params ([type]): Parameters where to input the model parameter values
        data_xs ([type]): Input data points
        data_ys ([type]): Class data points (0 or 1)
        initial_point ([type]): Initial set of parameters for the model
    Returns:
        model_values [list]: Optimal parameter values found by the optimizer
        loss [float]: Final cost value
        nfev [int]: Number of iteration done by the optimizer
    """

    target_values = 1 - 2*data_ys

    add_measurements = quantum_instance.backend_name != 'statevector_simulator'
    
    def cost_function(model_values):

        all_circuits = prepare_all_circuits(model_circuit,data_params,model_params,data_xs,model_values,add_measurements)
        all_results = quantum_instance.execute(all_circuits)
        expectation_values = all_counts_to_expectation_values(all_results.get_counts())
        all_costs = eval_cost_fct(expectation_values,target_values)
        return np.sum(all_costs)/len(all_costs)
    
    model_values, loss, nfev = optimizer.optimize(len(model_params), cost_function, initial_point=initial_point)

    return model_values, loss, nfev



def all_results_to_classifications(all_results):
    """Convert result into class
    Args:
        all_results ([type]): Results from running QuantumCircuits
    Returns:
        np.array: Prediction class (0 or 1)
    """
    
    expectation_values = all_results_to_expectation_values(all_results)
    classifications = np.choose(expectation_values>0,[1,0])

    return classifications


def classify(quantum_instance,model_circuit,model_params,model_values,data_params,data_xs):
    """Classify data point given a model, model values and a backend.
    Args:
        quantum_instance (Qiskit QuantumInstance): On which to run the QuantumCircuits.
        model_circuit (QuantumCircuit): The parametrized QuantumCircuit model.
        model_params ([type]): Parameters where to input the model parameter values
        model_values ([type]): Parameter values to be used into the model
        data_params ([type]): Parameters where to input the data
        data_xs ([type]): Input data points
    Returns:
        np.array: Prediction class (0 or 1)
    """

    add_measurements = quantum_instance.backend_name != 'statevector_simulator'

    all_circuits = prepare_all_circuits(model_circuit,data_params,model_params,data_xs,model_values,add_measurements)
    all_results = quantum_instance.execute(all_circuits)
    classifications = all_results_to_classifications(all_results)

    return classifications


# In[ ]:




