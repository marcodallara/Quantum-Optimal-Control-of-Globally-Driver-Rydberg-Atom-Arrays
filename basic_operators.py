import numpy as np
from scipy.sparse import kron
from scipy.sparse import csc_matrix
from scipy.sparse import eye
from scipy.sparse.linalg import expm
from qutip import Bloch, basis


def is_unitary(matrix):
    """
    Check if a given matrix is unitary.

    Parameters:
        matrix (ndarray): Matrix to check.

    Returns:
        bool: True if the matrix is unitary, False otherwise.
    """
    product = np.dot(matrix, np.conj(matrix.T))
    identity = np.eye(matrix.shape[0])
    return np.allclose(product, identity)


def gate_fidelity(U, U_target):
    """
    Calculate the gate fidelity between two unitary matrices.

    Parameters:
        U (ndarray): The unitary matrix representing the gate applied to the quantum state.
        U_target (ndarray): The target unitary matrix representing the desired gate operation.

    Returns:
        float: The gate fidelity, a measure of similarity between U and U_target.
    """
    dim = U.shape[0]
    return np.abs(np.trace(U_target.T.conj() @ U))**2 / dim**2


def build_submatrix(matrix, indices):
    """
    Build a submatrix of a given matrix.

    Parameters:
        matrix (ndarray): Matrix from which to extract the submatrix.
        indices (list): List of indices to extract from the matrix.

    Returns:
        ndarray: Submatrix of the given matrix.
    """
    return matrix[np.ix_(indices, indices)]


def H_Rabi(omega, phi, detuning=0, sparse=True):
    """
    Construct the Hamiltonian for a single qubit with Rabi oscillation driven by a laser.

    Parameters:
        omega (float): Rabi frequency of the laser.
        phi (float): Phase of the laser.
        detuning (float, optional): Detuning of the laser. Defaults to 0.
        sparse (bool, optional): If True, return a sparse matrix. Defaults to True.

    Returns:
        csr_matrix: Hamiltonian matrix.
    """
    H = (omega / 2) * (np.exp(1j * phi) * np.array([[0,1],[0,0]]) +
                       np.exp(-1j * phi) * np.array([[0,0],[1,0]]))
    H = H + detuning * np.array([[0,0],[0,1]])
    if sparse:
        H = csc_matrix(H)
    return H

def H_int_S(V, S):
    """
    Construct the total interaction Hamiltonian for S atoms all in blockade.

    Parameters:
        V (float): Rydberg interaction strength between the atoms.
        S (int): Number of atoms in a superatoms.

    Returns:
        csr_matrix: Interaction Hamiltonian matrix.
    """
    a = csc_matrix([[0, 0], [0, 1]])
    H = csc_matrix((2**S, 2**S), dtype=np.complex128)
    for i in range(S):
        for j in range(i+1, S):
            H += V*kron(kron(eye(2**i), kron(a, eye(2**(j-i-1)))), kron(a, eye(2**(S-j-1)))) 
    return H

def H_0_S(omega, phi, detuning, S):
    """
    Construct the total non-interaction Hamiltonian for S atoms all in blockade.

    Parameters:
        omega (float): Rabi frequency of the laser.
        phi (float): Phase of the laser.
        detuning (float): Detuning of the laser.
        S (int): Number of atoms in a superatoms.

    Returns:
        csr_matrix: Non-interaction Hamiltonian matrix.
    """
    a = H_Rabi(omega, phi, detuning)
    H = csc_matrix((2**S, 2**S), dtype=np.complex128)
    for i in range(S):
        H += kron(kron(eye(2**i), a), eye(2**(S-i-1)))
    return H

def time_evolution(sparse_hamiltonian, time):
    """
    Calculate the time evolution of a quantum system given a sparse Hamiltonian.

    Parameters:
        sparse_hamiltonian (csr_matrix): Sparse Hamiltonian matrix.
        time (float): Time at which to calculate the evolution.

    Returns:
        csr_matrix: Unitary evolution.
    """
    # Calculate the time evolution operator
    evolution_operator = expm(-1j * sparse_hamiltonian * time)

    return evolution_operator


def H_int_S_a(S, V):
    """
    Construct the interaction Hamiltonian between a superatom and an atom.

    Parameters:
        S (int): Number of atoms in a superatoms.
        V (float): Interaction strength between the qubits.

    Returns:
        csr_matrix: Interaction Hamiltonian matrix.
    """
    return H_int_S(V, S+1)

def H_int_S_S(S, V):
    """
    Construct the interaction Hamiltonian between a superatom and a superatom.

    Parameters:
        S (int): Number of atoms in a superatoms.
        V (float): Interaction strength between the qubits.

    Returns:
        csr_matrix: Interaction Hamiltonian matrix.
    """
    return H_int_S(V, 2*S)

def H_int_a_a(V):
    """
    Construct the interaction Hamiltonian between an atom and an atom.

    Parameters:
        V (float): Interaction strength between the qubits.

    Returns:
        csr_matrix: Interaction Hamiltonian matrix.
    """
    return H_int_S(V, 2)

def H_int_chain(chain, S, V):
    """
    Construct the interaction Hamiltonian for a chain of atoms.

    Parameters:
        chain (str): Chain configuration specifying the types of atoms.
        S (int): Number of atoms in a superatom.
        V (float): Interaction strength between the qubits.

    Returns:
        csr_matrix: Interaction Hamiltonian matrix.
    """
    # Calculate the number of atoms
    N = 0
    for i in range(0, len(chain)):
        if chain[i] == "A" or chain[i] == "B":
            N += 1
        elif chain[i] == "SA" or chain[i] == "SB":
            N += S

    # Initialize the interaction Hamiltonian
    H_int = csc_matrix((2**N, 2**N), dtype=np.complex128)

    j = 0 # Index of the atom in the chain
    for i in range(0, len(chain)-1):
        if chain[i] == "A":
            if chain[i+1] == "SA" or chain[i+1] == "SB":
                H_int += kron(eye(2**j), kron(H_int_S_a(S, V), eye(2**(N-j-1-S))))
            elif chain[i+1] == "B":
                H_int += kron(eye(2**j), kron(H_int_a_a(V), eye(2**(N-j-2))))
            j += 1

        elif chain[i] == "B":
            if chain[i+1] == "SA" or chain[i+1] == "SB":
                H_int += kron(eye(2**j), kron(H_int_S_a(S, V), eye(2**(N-j-1-S))))
            elif chain[i+1] == "A":
                H_int += kron(eye(2**j), kron(H_int_a_a(V), eye(2**(N-j-2))))
            j += 1

        elif chain[i] == "SA":
            if chain[i+1] == "A" or chain[i+1] == "B":
                H_int += kron(eye(2**j), kron(H_int_S_a(S, V), eye(2**(N-j-1-S))))
            elif chain[i+1] == "SB":
                H_int += kron(eye(2**j), kron(H_int_S_S(S, V), eye(2**(N-j-2*S))))
            j += S
            
        elif chain[i] == "SB":
            if chain[i+1] == "A" or chain[i+1] == "B":
                H_int += kron(eye(2**j), kron(H_int_S_a(S, V), eye(2**(N-j-1-S))))
            elif chain[i+1] == "SA":
                H_int += kron(eye(2**j), kron(H_int_S_S(S, V), eye(2**(N-j-2*S))))
            j += S

    return H_int


def H_0_chain(chain, omega_A, phi_A, detuning_A, omega_B, phi_B, detuning_B, S):
    """
    Construct the non-interaction Hamiltonian for a chain of atoms.

    Parameters:
        chain (str): Chain configuration specifying the types of atoms.
        omega_A (float): Rabi frequency of atom type A.
        phi_A (float): Phase of atom type A.
        detuning_A (float): Detuning of atom type A.
        omega_B (float): Rabi frequency of atom type B.
        phi_B (float): Phase of atom type B.
        detuning_B (float): Detuning of atom type B.
        S (int): Number of atoms in a superatom.

    Returns:
        csr_matrix: Non-interaction Hamiltonian matrix.
    """
    # Calculate the number of atoms
    N = 0
    for i in range(0, len(chain)):
        if chain[i] == "A" or chain[i] == "B":
            N += 1
        elif chain[i] == "SA" or chain[i] == "SB":
            N += S

    # Initialize the non-interaction Hamiltonian
    H_0 = csc_matrix((2**N, 2**N), dtype=np.complex128)
    
    j = 0 # Index of the atom in the chain
    for i in range(0, len(chain)):
        if chain[i] == "A":
            H_0 += kron(eye(2**j), kron(H_Rabi(omega_A, phi_A, detuning_A), eye(2**(N-j-1))))
            j += 1

        elif chain[i] == "B":
            H_0 += kron(eye(2**j), kron(H_Rabi(omega_B, phi_B, detuning_B), eye(2**(N-j-1))))
            j += 1

        elif chain[i] == "SA":
            H_0 += kron(eye(2**j), kron(H_0_S(omega_A, phi_A, detuning_A, S), eye(2**(N-j-S))))
            j += S

        elif chain[i] == "SB":
            H_0 += kron(eye(2**j), kron(H_0_S(omega_B, phi_B, detuning_B, S), eye(2**(N-j-S))))
            j += S

    return H_0
