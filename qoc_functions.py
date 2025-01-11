from basic_operators import *

def get_static_hamiltonian(chain, S, V, omega_A, phi_A, detuning_A, omega_B, phi_B, detuning_B):
    """
    Calculate the static part of the Hamiltonian for QOC for a chain of Rydberg atoms and superatoms.

    Parameters:
        chain (str): String representing the configuration of the quantum system.
        S (int): Number of atoms in a superatom.
        V (float): Rydberg Interaction.
        omega_A (float): Rabi frequency on species A.
        phi_A (float): Phase of laser on species A.
        detuning_A (float): Detuning on species A.
        omega_B (float): Rabi frequency on species B.
        phi_B (float): Phase of laser on species B.
        detuning_B (float): Detuning on species B.

    Returns:
        ndarray: Static part of the Hamiltonian.
    """
    H_s = H_int_chain(chain, S, V)
    H_s += H_0_chain(chain, omega_A, phi_A, detuning_A, omega_B, phi_B, detuning_B, S)
    return H_s

def H_control_A(ii):
    """
    Get the control Hamiltonian for a single atom.

    Parameters:
        ii (int): Index of the control Hamiltonian.

    Returns:
        ndarray: Control Hamiltonian matrix for a single atom.
    """
    if ii == 1:
        return  np.outer(np.array([[0], [1]]), np.array([[0], [1]]).conj())
    elif ii == 2:
        print("Invalid control Hamiltonian index")
    else:
        print("Invalid control Hamiltonian index")

def H_control_SA(ii, S):
    """
    Get the control Hamiltonian for a superatom.

    Parameters:
        ii (int): Index of the control Hamiltonian.
        S (int): Number of atoms in a superatom.

    Returns:
        ndarray: Control Hamiltonian matrix for a superatom.
    """
    H = np.zeros((2**S, 2**S))
    for i in range(S):
        H += kron(kron(eye(2**i), H_control_A(ii)), eye(2**(S-i-1)))
    return H


def get_control_hamiltonian_ii(chain, S, ii):
    """
    Get the control Hamiltonian for a specific index and configuration.

    Parameters:
        chain (str): String representing the configuration of the quantum system.
        S (int): Number of atoms in a superatom.
        ii (int): Index of the control Hamiltonian.

    Returns:
        ndarray: Control Hamiltonian for the given index and configuration.
    """
    N = 0
    for i in range(0, len(chain)):
        if chain[i] == "A" or chain[i] == "B":
            N += 1
        elif chain[i] == "SA" or chain[i] == "SB":
            N += S
    H_0 = 0
    j = 0 
    for i in range(0, len(chain)):
        if chain[i] == "B":
            H_0 += kron(eye(2**j), kron(H_control_A(ii), eye(2**(N-j-1))))
            j += 1

        elif chain[i] == "SB":
            H_0 += kron(eye(2**j), kron(H_control_SA(ii, S), eye(2**(N-j-S))))
            j += S

        if chain[i] == "A":
            j += 1

        elif chain[i] == "SA":
            j += S
    return H_0

def time_evolution_qoc(props, drive, H_drift, H_control, n_slices, dt):
    """
    Calculate the time evolution of a quantum system given a sparse Hamiltonian.

    Parameters:
        props (list): List of propagators where results will be stored.
        drive (list): List of drive signals.
        H_drift (ndarray): Drift Hamiltonian.
        H_control (list): List of control Hamiltonians.
        n_slices (int): Number of time slices.
        dt (float): Time step.

    Returns:
        list: Unitary evolution.
    """
    for i in range(n_slices):
        props[i] = expm(-1j * (H_drift + drive[0][i]*H_control[0]) * dt)
    return props
