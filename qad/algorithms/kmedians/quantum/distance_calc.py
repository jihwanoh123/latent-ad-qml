import numpy as np
from qibo.models import Circuit
from qibo import gates
from qad.algorithms.kmedians.util import calc_norm
#######Modified##########
from qiskit import QuantumCircuit
from qiskit_ionq import IonQProvider
import math
########################
def pad_input(X):
    """Adds 0s if X log2(X.dim) != round int.

    Parameters
    ----------
    X : `numpy.ndarray`
        Input data

    Returns
    -------
    `numpy.ndarray`
        Padded X
    """
    num_features = len(X)
    if not float(np.log2(num_features)).is_integer():
        size_needed = pow(2, math.ceil(math.log(num_features) / math.log(2)))
        X = np.pad(X, (0, size_needed - num_features), "constant")
    return X


def DistCalc_DI(a, b, device_name='ionq_simulator',
                #"/GPU:0",
                shots_n=10000):
    """Distance calculation using destructive interference.

    Parameters
    ----------
    a : :class:`numpy.ndarray`
        First point - shape = (latent space dimension,)
    b : :class:`numpy.ndarray`
        First point - shape = (latent space dimension,)
    device_name : str
        Name of device for executing a simulation of quantum circuit.
    shots_n : int
        Number of shots for executing a quantum circuit - to get frequencies.

    Returns
    -------
    (float, :class:`qibo.models.Circuit`)
        (distance, quantum circuit)

    """
    num_features = len(a)
    norm = calc_norm(a, b)
    a_norm = a / norm
    b_norm = b / norm

    a_norm = pad_input(a_norm)
    b_norm = pad_input(b_norm)

    amplitudes = np.concatenate((a_norm, b_norm))
    n_qubits = int(np.log2(len(amplitudes)))

    # QIBO
    #qc = Circuit(n_qubits)
    #qc.add(gates.H(0))
    #qc.add(gates.M(0))
    #with tf.device(device_name):
    #    result = qc.execute(initial_state=amplitudes, nshots=shots_n)
    #counts = result.frequencies(binary=True)
    #distance = norm * math.sqrt(2) * math.sqrt((counts["1"] / shots_n))

    #IonQ
    provider = IonQProvider("27pEbHsUIEAy6WtHdddSLF8UjtwlA4fR")
    simulator_backend = provider.get_backend(device_name)
    qc = QuantumCircuit(n_qubits,n_qubits)
    qc.h(0)
    qc.measure(0,0)
    #Run the circuit on IonQ's platform:
    job = simulator_backend.run(qc,shots=shots_n)
    counts = job.get_counts()
    distance = norm * math.sqrt(2) * math.sqrt((counts["001"] / shots_n))
    
    return distance, qc
