# Main script of the qsvm.
# Imports the data for training. Imports the data for validation and testing
# and kfolds it into k=5.
# Computes the ROC curve of the qsvm and the AUC, saves the ROC plot.
import warnings
from time import perf_counter

from qiskit.utils import algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel
import numpy as np

from sklearn.svm import SVC
from sklearn import metrics

from .terminal_colors import tcols
from . import qdata as qd
from . import util
from . import test
from .feature_map_circuits import u2Reuploading

# Warnings are suppressed since qiskit aqua obfuscates the output of this
# script otherwise (IBM's fault not ours.)
# warnings.filterwarnings("ignore", category=DeprecationWarning)

seed = 12345 
# Ensure same global behaviour.
algorithm_globals.random_seed = seed


def main(args):
    train_loader, test_loader = util.get_data(args)
    train_features, train_labels = train_loader[0], train_loader[1]
    test_features, test_labels = test_loader[0], test_loader[1]

    feature_map = u2Reuploading(nqubits=8)
    quantum_instance, backend = util.configure_quantum_instance(
        ibmq_api_config=args["ibmq_api_config"],
        run_type=args["run_type"],
        backend_name=args["backend_name"],
        **args["config"],
    )
    kernel = QuantumKernel(feature_map=feature_map, 
                           quantum_instance=quantum_instance)
    print("Calculating the quantum kernel matrix elements... ", end="")
    train_time_init = perf_counter()
    quantum_kernel_matrix = kernel.evaluate(x_vec=train_features)
    train_time_fina = perf_counter()
    print(
        tcols.OKGREEN + f"Done in: {train_time_fina-train_time_init:.2e} s" + tcols.ENDC
    )

    qsvm = SVC(kernel="precomputed", C=args["c_param"])
    out_path = util.create_output_folder(args, qsvm)
    np.save(out_path + "/kernel_matrix_elements", quantum_kernel_matrix)

    print("Training the QSVM...", end="")
    train_time_init = perf_counter()
    qsvm.fit(quantum_kernel_matrix, train_labels)
    train_time_fina = perf_counter()
    print(f"Training completed in: {train_time_fina-train_time_init:.2e} s")
    util.print_model_info(qsvm)

    print(tcols.OKCYAN + "\nTesting the accuracy of the models..." + tcols.ENDC)
    kernel_matrix_test = kernel.evaluate(x_vec=test_features, y_vec=train_features)
    train_acc = qsvm.score(quantum_kernel_matrix, train_labels)
    test_acc = qsvm.score(kernel_matrix_test, test_labels)
    util.print_auc_scores(test_acc, train_acc)
    util.save_qsvm(qsvm, out_path + "/model")
    qc_transpiled = util.get_quantum_kernel_circuit(kernel, out_path)
    
    # TODO add flag to check if I want to do test+train in one go?

    if backend is not None:
        util.save_circuit_physical_layout(qc_transpiled, backend, out_path)
        util.save_backend_properties(backend, out_path + "/backend_properties_dict")
