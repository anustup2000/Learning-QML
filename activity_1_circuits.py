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

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter

def build_data_embedding_circuit():
    """Builds a plain 2D data embedding circuit
    Returns:
        QuantumCircuit: The data embedding parametrized quantum circuit
        ParameterVector or list[Parameter] : The parameters to be used to embed the data
    """

    data_params = ParameterVector('x', 2)
    
    data_embedding_circuit = QuantumCircuit(1)

    data_embedding_circuit.ry(data_params[0],0)
    data_embedding_circuit.rz(data_params[1],0)

    return data_embedding_circuit, data_params

def build_rotation_model_circuit():
    """Builds the rotation model quantum circuit. 
    First embeds the data.
    Then rotates the data.
    Returns:
        QuantumCircuit: The rotation model quantum circuit
        ParameterVector or list[Parameter] : The parameters to be used to embed the data
        ParameterVector or list[Parameter] : The model's parameters
    """

    data_params = ParameterVector('x', 2)
    rotation_params = ParameterVector('m', 2)

    model_circuit = QuantumCircuit(1)
    model_circuit.ry(data_params[0],0)
    model_circuit.rz(data_params[1],0)
    
    model_circuit.rz(rotation_params[0],0)
    model_circuit.ry(rotation_params[1],0)
    
    return model_circuit, data_params, rotation_params

def build_linear_model_circuit():
    """Builds the linear model quantum circuit. 
    First embeds the data.
    Then rotates the data.
    Returns:
        QuantumCircuit: The linear model quantum circuit
        ParameterVector or list[Parameter] : The parameters to be used to embed the data
        ParameterVector or list[Parameter] : The model's parameters (includes rotations and weights)
    """

    data_params = ParameterVector('x', 2)
    weights_params = ParameterVector('w', 2)
    rotation_params = ParameterVector('m', 2)

    model_params = list(rotation_params) + list(weights_params)
    
    model_circuit = QuantumCircuit(1)
    model_circuit.ry(weights_params[0] * data_params[0],0)
    model_circuit.rz(weights_params[1] * data_params[1] + rotation_params[1],0)
    model_circuit.ry(rotation_params[0],0)
    
    return model_circuit, data_params, model_params

def build_layered_model_circuit(n_layers = 1):
    """Builds the layered model quantum circuit. 
    Takes care of the weighted data embedding and the rotations on many layers.
    Args:
        n_layers (int, optional): The number of layers. Defaults to 1.
    Returns:
        QuantumCircuit: The layered model quantum circuit
        ParameterVector or list[Parameter] : The parameters to be used to embed the data
        ParameterVector or list[Parameter] : The model's parameters (includes rotations and weights)
    """

    data_params = ParameterVector('x', 2)
    weights_params = ParameterVector('w', 2*n_layers)
    rotation_params = ParameterVector('m', 2*n_layers)

    model_params = list(rotation_params) + list(weights_params)
    
    model_circuit = QuantumCircuit(1)
    model_circuit.ry(weights_params[0] * data_params[0],0)
    model_circuit.rz(weights_params[1] * data_params[1] + rotation_params[1],0)
    for l in range(1,n_layers):
        model_circuit.ry(weights_params[2*l+0] * data_params[0] + rotation_params[2*l+0],0)
        model_circuit.rz(weights_params[2*l+1] * data_params[1] + rotation_params[2*l+1],0)
    model_circuit.ry(rotation_params[0],0)
    
    return model_circuit, data_params, model_params


# In[ ]:




