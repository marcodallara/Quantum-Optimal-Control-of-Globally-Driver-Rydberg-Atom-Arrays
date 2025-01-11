from basic_operators import *
from qoc_functions import *
from functools import reduce
from Chain import Chain
import time 

class Grid:
    def __init__(self, chain_list, SB_list, S, V, psi_0, U_list=[]):
        """
        Constructor method.
        
        Parameters:
            S: int - Number of atoms in a superatom (SA or SB).
            V: float - Interaction strength.
            psi_0: numpy array - Initial state of the system.
            U_list: list, optional - List of unitary operators. Defaults to empty list.
        """
        # Initialize class attributes
        self.SB_list = SB_list  # Represents the list of coords of SB imperfections. e.g. [(0,1)] means one imperfections between rows 0 and 1 and and column 1
        self.S = S  # Represents the number of atoms in a superatom (SA or SB)
        self.V = V  # Represents the interaction strength
        self.psi_0 = psi_0  # Represents the initial state of the system
        self.psi = psi_0  # Represents the current state of the system
        self.chain_list = chain_list  # Represents the list of chains of atoms or superatoms
        self.N_list = [chain_list[i].get_N() for i in range(len(chain_list))]  # Represents the number of atoms in the chains
        # Calculate the total number of atoms in the chain
        self.N_tot_chain = sum(self.N_list)
        self.N_SB = S * len(SB_list)  # Represents the number of atoms in the SB imperfections
        self.N_grid = self.N_tot_chain + self.N_SB
        self.n_slices = 100  # Represents the number of slices for time evolution
        self.N_op = 0  # Represents the number of gate operations
        self.psi_t = [self.psi_0]  # Represents the list of states of the system at different times

    def Z_tot(self, omega_B, phi_B):
        """
        Calculate the total Z operator.

        Parameters:
            omega_B: float - Omega parameter for the Hamiltonian.
            phi_B: float - Phi parameter for the Hamiltonian.

        Returns:
            numpy array - The total Z operator.
        """    
        H_list = []
        # Compute the total Z operator for the system
        
        # print the time for each bloch of code
        for chain in self.chain_list:   
            chain_string = chain.get_chain()
            chain_H_int = H_int_chain(chain_string, chain.get_S(), chain.get_V())
            chain_H_0 = H_0_chain(chain_string, 0, 0, 0, omega_B, phi_B, 0, chain.get_S())
            chain_H = chain_H_0 + chain_H_int
            H_list.append(chain_H)

        H_final = csc_matrix((2**self.N_grid, 2**self.N_grid), dtype=np.complex128)
        H_final = kron(H_list[0], eye(2**(self.N_grid - self.N_list[0])))
        csum = np.cumsum(self.N_list)
        for i in range(1, len(H_list)):
            H_final += kron(eye(2**(csum[i-1])), kron(H_list[i], eye(2**(self.N_grid - csum[i]))))
        H_final = csc_matrix(H_final)

        t = 2 * np.pi / omega_B 
        # Calculate the total Z operator for the system

        U = time_evolution(H_final, t)

        return U  # Return the total Z operator


    def Hadamard(self, omega_A, phi_A, omega_B, phi_B):
        """
        Apply a Hadamard gate to the quantum system.

        Parameters:
            omega_A (float): Rabi frequency of atom type A.
            phi_A (float): Phase of atom type A.
            omega_B (float): Rabi frequency of atom type B.
            phi_B (float): Phase of atom type B.

        Returns:
            ndarray: Updated quantum state after applying the Hadamard gate.
        """
        # Calculate the total operator for the Hadamard gate
        phi_A = 0
        t = np.pi/4 / omega_A
        H_list = []
        for chain in self.chain_list:   
            chain_string = chain.get_chain()
            chain_H_int = H_int_chain(chain_string, chain.get_S(), chain.get_V())
            chain_H_0 = H_0_chain(chain_string, omega_A, phi_A, 0, 0, 0, 0, chain.get_S())
            chain_H = chain_H_0 + chain_H_int
            H_list.append(chain_H)

        H_final = csc_matrix((2**self.N_grid, 2**self.N_grid), dtype=np.complex128)
        H_final = kron(H_list[0], eye(2**(self.N_grid - self.N_list[0])))
        csum = np.cumsum(self.N_list)
        for i in range(1, len(H_list)):
            H_final += kron(eye(2**(csum[i-1])), kron(H_list[i], eye(2**(self.N_grid - csum[i]))))
        H_final = csc_matrix(H_final)
        U1 = time_evolution(H_final, t)



        phi_A = np.pi/2
        t = np.pi/2 / omega_A
        H_list = []
        for chain in self.chain_list:   
            chain_string = chain.get_chain()
            chain_H_int = H_int_chain(chain_string, chain.get_S(), chain.get_V())
            chain_H_0 = H_0_chain(chain_string, omega_A, phi_A, 0, 0, 0, 0, chain.get_S())
            chain_H = chain_H_0 + chain_H_int
            H_list.append(chain_H)

        H_final = csc_matrix((2**self.N_grid, 2**self.N_grid), dtype=np.complex128)
        H_final = kron(H_list[0], eye(2**(self.N_grid - self.N_list[0])))
        csum = np.cumsum(self.N_list)
        for i in range(1, len(H_list)):
            H_final += kron(eye(2**(csum[i-1])), kron(H_list[i], eye(2**(self.N_grid - csum[i]))))
        H_final = csc_matrix(H_final)
        U2 = time_evolution(H_final, t)


        phi_A = -np.pi/4
        t = np.pi*3/4 / omega_A
        H_list = []
        for chain in self.chain_list:   
            chain_string = chain.get_chain()
            chain_H_int = H_int_chain(chain_string, chain.get_S(), chain.get_V())
            chain_H_0 = H_0_chain(chain_string, omega_A, phi_A, 0, 0, 0, 0, chain.get_S())
            chain_H = chain_H_0 + chain_H_int
            H_list.append(chain_H)

        H_final = csc_matrix((2**self.N_grid, 2**self.N_grid), dtype=np.complex128)
        H_final = kron(H_list[0], eye(2**(self.N_grid - self.N_list[0])))
        csum = np.cumsum(self.N_list)
        for i in range(1, len(H_list)):
            H_final += kron(eye(2**(csum[i-1])), kron(H_list[i], eye(2**(self.N_grid - csum[i]))))
        H_final = csc_matrix(H_final)
        U3 = time_evolution(H_final, t)

        phi_A = np.pi/4
        t = np.pi*3/2 / omega_A
        H_list = []
        for chain in self.chain_list:   
            chain_string = chain.get_chain()
            chain_H_int = H_int_chain(chain_string, chain.get_S(), chain.get_V())
            chain_H_0 = H_0_chain(chain_string, omega_A, phi_A, 0, 0, 0, 0, chain.get_S())
            chain_H = chain_H_0 + chain_H_int
            H_list.append(chain_H)

        H_final = csc_matrix((2**self.N_grid, 2**self.N_grid), dtype=np.complex128)
        H_final = kron(H_list[0], eye(2**(self.N_grid - self.N_list[0])))
        csum = np.cumsum(self.N_list)
        for i in range(1, len(H_list)):
            H_final += kron(eye(2**(csum[i-1])), kron(H_list[i], eye(2**(self.N_grid - csum[i]))))
        H_final = csc_matrix(H_final)
        U4 = time_evolution(H_final, t)

        U = U4 @ U3 @ U2 @ U1
        U_dag = U.conj().T
        Ztot = self.Z_tot(omega_B, phi_B)
        return U_dag@Ztot@U  # Return the total operator for the Hadamard gate
    

    #def CZ(self, omega_B, phi_B, props=False):  TO IMPLEMENT



    def Hadamard_S2(self, omega_A, omega_B, phi_B, N=2, params=[2.4115961, 3.14159335, 2.98995701, 3.14159277]):
        """
        Apply a sequence of Hadamard gates to the quantum state with S=2 manually optimized.

        Parameters:
            omega_A (float): Rabi frequency of the Hadamard gate on qubit A.
            omega_B (float): Rabi frequency of the Hadamard gate on qubit B.
            phi_B (float): Phase of the Hadamard gate on qubit B.
            props (bool, optional): If True, returns the unitary operator representing the gate operation. Defaults to False.
            N (int, optional): Number of concatenated Hadamard gates. Defaults to 2.
            params (list, optional): List of parameters for the concatenated Hadamard gates. 
                                    Each pair of values in the list represents phi_A and t/omega_A for each gate. 
                                    Defaults to [2.4115961, 3.14159335, 2.98995701, 3.14159277].

        Returns:
            ndarray or csr_matrix: The resulting quantum state after applying the Hadamard gate sequence if props is False.
                                The unitary operator representing the gate operation if props is True.
        """

        # Apply N concatenated Hadamard gates
        for j in range(N):
            phi_A, t = params[j*2], (params[j*2+1] / omega_A)
            
            # Calculate U for each gate
            H_list = []
            # Compute the total Z operator for the system
            for chain in self.chain_list:   
                chain_string = chain.get_chain()
                chain_H_int = H_int_chain(chain_string, chain.get_S(), chain.get_V())
                chain_H_0 = H_0_chain(chain_string, omega_A, phi_A, 0, 0, 0, 0, chain.get_S())
                chain_H = chain_H_0 + chain_H_int
                H_list.append(chain_H)
            H_final = csc_matrix((2**self.N_grid, 2**self.N_grid), dtype=np.complex128)
            H_final = kron(H_list[0], eye(2**(self.N_grid - self.N_list[0])))
            csum = np.cumsum(self.N_list)
            for i in range(1, len(H_list)):
                H_final += kron(eye(2**(csum[i-1])), kron(H_list[i], eye(2**(self.N_grid - csum[i]))))
            H_final = csc_matrix(H_final)
            Uj = time_evolution(H_final, t)
            if j == 0:
                U = Uj
            else:
                U = Uj @ U

        # Calculate the adjoint of U
        U_dag = U.conj().T
        
        # Calculate the total Z operator
        Ztot = self.Z_tot(omega_B, phi_B)
        
        return U_dag @ Ztot @ U

"""next ones still need to be implemented"""

    # def CZ_S2(self, omega_B, phi_B, params=None):
    #     """
    #     Implements the controlled-Z (CZ) gate operation of the paper arXiv:2305.19220v2 adapted to S=2.

    #     Parameters:
    #         omega_B (float): Rabi frequency of the control qubit.
    #         phi_B (float): Phase of the control qubit.
    #         props (bool): If True, returns the total operator for the CZ gate; 
    #                     if False, applies the CZ gate operation to the quantum state.
    #         params (array-like, optional): Parameters for the CZ gate operation.

    #     Returns:
    #         ndarray or None: If props is True, returns the total operator for the CZ gate;
    #                         if props is False, returns None.
    #     """
    #     if params is None:
    #         params = np.array([np.pi/2, np.pi/4, 0, np.pi, np.pi/2, np.pi/2, 0, np.pi, np.pi/2, np.pi/4])
    #         params[1::2] *= 2 / np.sqrt(self.S)  # Scale times for different superatom values

    #     N = len(params) // 2  # Number of concatenated pulses

    #     for j in range(N):
    #         phi_B, t = params[j*2], (params[j*2+1] / omega_B)
    #         # Calculation of U   
    #         # Calculate U for each gate
    #         H_list = []
    #         # Compute the total Z operator for the system
    #         for chain in self.chain_list:   
    #             chain_string = chain.get_chain()
    #             chain_H_int = H_int_chain(chain_string, chain.get_S(), chain.get_V())
    #             chain_H_0 = H_0_chain(chain_string, 0, 0, 0, omega_B, phi_B, 0, chain.get_S())
    #             chain_H = chain_H_0 + chain_H_int
    #             H_list.append(chain_H)
    #         H_final = csc_matrix((2**self.N_grid, 2**self.N_grid), dtype=np.complex128)
    #         H_final = kron(H_list[0], eye(2**(self.N_grid - self.N_list[0])))
    #         csum = np.cumsum(self.N_list)
    #         for i in range(1, len(H_list)):
    #             H_final += kron(eye(2**(csum[i-1])), kron(H_list[i], eye(2**(self.N_grid - csum[i]))))
    #         H_final = csc_matrix(H_final)
    #         # Add interatction between the chains


    #         Uj = time_evolution(H_final, t)
    #         if j == 0:
    #             U = Uj
    #         else:
    #             U = Uj @ U

    #     return U



    # def CZ_qoc(self, omega_B, phi_B, props=False, pulse=None, timegrid0=None):
    #     """
    #     Implements the controlled-Z (CZ) gate operation using quantum optimal control: dCRAB with symmetric pulses.

    #     Parameters:
    #         omega_B (float): Rabi frequency of the control qubit.
    #         phi_B (float): Phase of the control qubit.
    #         props (bool): If True, returns the total operator for the CZ gate;
    #                     if False, applies the CZ gate operation to the quantum state.
    #         pulse (array-like, optional): Custom pulse shape for the control qubit.
    #         timegrid0 (array-like, optional): Custom time grid for the pulse.

    #     Returns:
    #         ndarray or None: If props is True, returns the total operator for the CZ gate;
    #                         if props is False, returns None.
    #     """
    #     if pulse is None:
    #         pulse0 = list(np.load("pulse0.npy"))
    #         timegrid0 = list(np.load("timegrid0.npy"))
    #     else:
    #         pulse0 = pulse
    #         timegrid0 = timegrid0
        
    #     n_slices = 100
    #     H_control = [get_control_hamiltonian_ii(self.chain, self.S, 1).toarray()]
    #     H_drift = get_static_hamiltonian(self.chain, self.S, self.V, 0, 0, 0, omega_B, 0, 0).toarray()
    #     pulses = [pulse0]
    #     timegrid = [timegrid0]
    #     dt = timegrid0[1] - timegrid0[0]
    #     props_0 = [np.eye(2**self.N, dtype=np.complex128) for i in range(n_slices)]
    #     U = time_evolution_qoc(props_0, pulses, H_drift, H_control, 100, dt)
        
    #     if props:
    #         U_1 = reduce(lambda a, b: a @ b, U)
    #     else: 
    #         for i in range(n_slices):
    #             self.psi = U[n_slices-i-1] @ self.psi
    #             self.psi_t.append(self.psi)

    #     flipped_pulses = [np.flip(pulse0)]
    #     props_0 = [np.eye(2**self.N, dtype=np.complex128) for i in range(n_slices)]
    #     U = time_evolution_qoc(props_0, flipped_pulses, H_drift, H_control, 100, dt)
        
    #     if props:
    #         U_2 = reduce(lambda a, b: a @ b, U)
    #     else:
    #         for i in range(n_slices):
    #             self.psi = U[n_slices-i-1] @ self.psi
    #             self.psi_t.append(self.psi)
            
    #     if props:
    #         return U_2 @ U_1
    #     else:
    #         self.N_op += 2
    #         return self.psi
        
    # def Bloch_sphere_plot(self, z_labels, indexes_0, indexes_1, angles=[90, 15], filename=None):
    #     """
    #     Print the evolution of the state in a Bloch sphere.

    #     Parameters:
    #         z_labels: list - List of labels for the z-axis.
    #         indexes_0: list - List of indices for the first basis vector.
    #         indexes_1: list - List of indices for the second basis vector.
    #         angles: list, optional - Viewing angles for the Bloch sphere. Defaults to [90, 15].
    #     """
    #     b = Bloch(view=angles)
    #     b.point_marker = ['o']
    #     colors = ['r']
    #     colors += ['b' for i in range(self.n_slices*self.N_op - 5)]
    #     for i in range(5):
    #         colors.append('g')
    #     b.point_color = colors
    #     b.point_size = [15]
    #     b.font_size = 18
    #     b.zlpos = [1.2, -1.2]
    #     b.zlabel = z_labels
    #     for psi_i in self.psi_t:
    #         c_0 = np.sum(psi_i[indexes_0])/np.sqrt(len(indexes_0))
    #         c_1 = np.sum(psi_i[indexes_1])/np.sqrt(len(indexes_1))
    #         vec = (c_0 * basis(2, 0) + c_1 * basis(2, 1))
    #         b.add_states(vec, kind='point')

    #     b.figsize = [5, 10]
    #     b.render()
    #     # save with the filename
    #     if filename is not None:
    #         b.save(filename)
    #         print("Bloch sphere plot saved as", filename)
    #     else:
    #         b.save()
