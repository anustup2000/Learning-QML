import sys


def stp():
    sys.stdout.write("#")
    sys.stdout.flush()


toolbar_width = 17
progress = "Progress: "
sys.stdout.write(progress + "[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

try:

    import numpy as np
    stp()
    import matplotlib.pyplot as plt
    stp()
    import numpy as np
    stp()
    import os
    stp()
    from qiskit import QuantumCircuit, Aer, execute, IBMQ, transpile
    stp()
    from qiskit.algorithms.optimizers import COBYLA, SPSA
    stp()
    from qiskit.circuit import Parameter
    stp()
    from qiskit.circuit.library import TwoLocal, ZZFeatureMap
    stp()
    from qiskit_machine_learning.datasets import ad_hoc_data
    stp()
    from qiskit_machine_learning.algorithms.classifiers import VQC
    stp()
    from qiskit.utils import QuantumInstance
    stp()
    from qiskit.visualization import plot_bloch_multivector
    stp()
    from qiskit.visualization.utils import _bloch_multivector_data
    stp()
    from qiskit.visualization.bloch import Bloch
    stp()
    from sklearn import datasets
    stp()
    from sklearn.model_selection import train_test_split
    stp()
    from skimage.transform import resize
    stp()
    import plotly.graph_objects as go

    sys.stdout.write("]\n")
    print("\nCongratulations, your environment is configured correctly.\nHappy QML workshop!\n")

except ModuleNotFoundError as e:
    print("\n" + "@"*80 + "\n")
    print("Your environment is missing the following dependency:")
    print(f' *  {e}\n')
    print("@"*80)
