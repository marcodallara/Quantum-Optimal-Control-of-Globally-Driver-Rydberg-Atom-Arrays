from basic_operators import *
from qoc_functions import *
from functools import reduce


class Chain:
    def __init__(self, chain, S, V, psi_0, U_list=[]):
        """
        Constructor method.
        
        Parameters:
            chain: str - Chain of atoms or superatoms.
            S: int - Number of atoms in a superatom (SA or SB).
            V: float - Interaction strength.
            psi_0: numpy array - Initial state of the system.
            U_list: list, optional - List of unitary operators. Defaults to empty list.
        """
        # Initialize class attributes
        self.chain = chain  # Represents the chain of atoms or superatoms
        self.S = S  # Represents the number of atoms in a superatom (SA or SB)
        self.V = V  # Represents the interaction strength
        self.psi_0 = psi_0  # Represents the initial state of the system
        self.psi = psi_0  # Represents the current state of the system
        self.N = 0  # Represents the number of atoms in the chain
        # Calculate the total number of atoms in the chain
        for i in range(0, len(chain)):
            if chain[i] == "A" or chain[i] == "B":
                self.N += 1
            elif chain[i] == "SA" or chain[i] == "SB":
                self.N += S
        self.n_slices = 100  # Represents the number of slices for time evolution
        self.N_op = 0  # Represents the number of gate operations
        self.psi_t = [self.psi_0]  # Represents the list of states of the system at different times

    def get_N(self):
        """
        Get the number of atoms in the chain.

        Returns:
            int: The number of atoms in the chain.
        """
        return self.N
    
    def get_S(self):
        """
        Get the number of atoms in a superatom.

        Returns:
            int: The number of atoms in a superatom.
        """
        return self.S
    
    def get_V(self):
        """
        Get the interaction strength.

        Returns:
            float: The interaction strength.
        """
        return self.V
    
    def get_chain(self):
        """
        Get the chain of atoms or superatoms.

        Returns:
            str: The chain of atoms or superatoms.
        """
        return self.chain

    def Z_tot(self, omega_B, phi_B, props=False):
        """
        Calculate the total Z operator.

        Parameters:
            omega_B: float - Omega parameter for the Hamiltonian.
            phi_B: float - Phi parameter for the Hamiltonian.

        Returns:
            numpy array - The total Z operator.
        """            
        # Compute the total Z operator for the system
        chain_H_int = H_int_chain(self.chain, self.S, self.V)
        chain_H_0 = H_0_chain(self.chain, 0, 0, 0, omega_B, phi_B, 0, self.S)
        chain_H = chain_H_0 + chain_H_int
        t = 2 * np.pi / omega_B 
        if props == True:
            # Calculate the total Z operator for the system
            U = time_evolution(chain_H, t)
            return U  # Return the total Z operator

        else:
            # Perform time evolution
            dt = t / self.n_slices
            U = time_evolution(chain_H, dt)
            for i in range(self.n_slices):
                self.psi = U @ self.psi
                self.psi_t.append(self.psi)
            self.N_op += 1
            return self.psi  # Return the total Z operator


    def Hadamard(self, omega_A, phi_A, omega_B, phi_B, props=False):
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
        if props == True:
            # Calculate the total operator for the Hadamard gate
            phi_A = 0
            t = np.pi/4 / omega_A
            chain_H_int = H_int_chain(self.chain, self.S, self.V)
            chain_H_0 = H_0_chain(self.chain, omega_A, phi_A, 0, 0, 0, 0, self.S)
            chain_H = chain_H_0 + chain_H_int
            U1 = time_evolution(chain_H, t)

            phi_A = np.pi/2
            t = np.pi/2 / omega_A
            chain_H_0 = H_0_chain(self.chain, omega_A, phi_A, 0, 0, 0, 0, self.S)
            chain_H = chain_H_0 + chain_H_int
            U2 = time_evolution(chain_H, t)

            phi_A = -np.pi/4
            t = np.pi*3/4 / omega_A
            chain_H_0 = H_0_chain(self.chain, omega_A, phi_A, 0, 0, 0, 0, self.S)
            chain_H = chain_H_0 + chain_H_int
            U3 = time_evolution(chain_H, t)

            phi_A = np.pi/4
            t = np.pi*3/2 / omega_A
            chain_H_0 = H_0_chain(self.chain, omega_A, phi_A, 0, 0, 0, 0, self.S)
            chain_H = chain_H_0 + chain_H_int
            U4 = time_evolution(chain_H, t)

            U = U4 @ U3 @ U2 @ U1
            U_dag = U.conj().T
            Ztot = self.Z_tot(omega_B, phi_B, props=True)
            return U_dag@Ztot@U  # Return the total operator for the Hadamard gate

        else:
            # Reset phi_A to 0 for the first part of the Hadamard gate
            phi_A = 0
            # Define time and time step for the first part of the gate
            t = np.pi/4 / omega_A
            dt = t / (self.n_slices)
            # Construct interaction and non-interaction Hamiltonians
            chain_H_int = H_int_chain(self.chain, self.S, self.V)
            chain_H_0 = H_0_chain(self.chain, omega_A, phi_A, 0, 0, 0, 0, self.S)
            chain_H = chain_H_0 + chain_H_int
            # Perform time evolution
            U1 = time_evolution(chain_H, dt)
            for i in range(self.n_slices):
                self.psi = U1 @ self.psi
                self.psi_t.append(self.psi)

            # Set phi_A to pi/2 for the second part of the Hadamard gate
            phi_A = np.pi/2
            # Define time and time step for the second part of the gate
            t = np.pi/2 / omega_A
            dt = t / (self.n_slices)
            # Reconstruct non-interaction Hamiltonian
            chain_H_0 = H_0_chain(self.chain, omega_A, phi_A, 0, 0, 0, 0, self.S)
            chain_H = chain_H_0 + chain_H_int
            # Perform time evolution
            U2 = time_evolution(chain_H, dt)
            for i in range(self.n_slices):
                self.psi = U2 @ self.psi
                self.psi_t.append(self.psi)
            
            # Set phi_A to -pi/4 for the third part of the Hadamard gate
            phi_A = -np.pi/4
            # Define time and time step for the third part of the gate
            t = np.pi*3/4 / omega_A
            dt = t / (self.n_slices)
            # Reconstruct non-interaction Hamiltonian
            chain_H_0 = H_0_chain(self.chain, omega_A, phi_A, 0, 0, 0, 0, self.S)
            chain_H = chain_H_0 + chain_H_int
            # Perform time evolution
            U3 = time_evolution(chain_H, dt)
            for i in range(self.n_slices):
                self.psi = U3 @ self.psi
                self.psi_t.append(self.psi)
            
            # Set phi_A to pi/4 for the fourth part of the Hadamard gate
            phi_A = np.pi/4
            # Define time and time step for the fourth part of the gate
            t = np.pi*3/2 / omega_A
            dt = t / (self.n_slices)
            # Reconstruct non-interaction Hamiltonian
            chain_H_0 = H_0_chain(self.chain, omega_A, phi_A, 0, 0, 0, 0, self.S)
            chain_H = chain_H_0 + chain_H_int
            # Perform time evolution
            U4 = time_evolution(chain_H, dt)
            for i in range(self.n_slices):
                self.psi = U4 @ self.psi
                self.psi_t.append(self.psi)

            # Apply Z_tot gate with parameters omega_B and phi_B
            _ = self.Z_tot(omega_B, phi_B)

            # Take the conjugate transpose of each evolution operator
            U1_dag = U1.conj().T
            U2_dag = U2.conj().T
            U3_dag = U3.conj().T
            U4_dag = U4.conj().T

            # Undo the evolution by applying the conjugate transpose of each evolution operator in reverse order
            for i in range(self.n_slices):
                self.psi = U4_dag @ self.psi
                self.psi_t.append(self.psi)
            
            for i in range(self.n_slices):
                self.psi = U3_dag @ self.psi
                self.psi_t.append(self.psi)

            for i in range(self.n_slices):
                self.psi = U2_dag @ self.psi
                self.psi_t.append(self.psi)

            for i in range(self.n_slices):
                self.psi = U1_dag @ self.psi
                self.psi_t.append(self.psi)

            # Update the number of operators applied
            self.N_op += 8
            return self.psi

    def CZ(self, omega_B, phi_B, props=False):
        """
        Implements the CZ gate operation on the quantum state as in the paper arXiv:2305.19220v2, with S=4.

        Parameters:
            omega_B (float): Rabi frequency of the control laser for qubit B.
            phi_B (float): Phase of the control laser for qubit B.
            props (bool, optional): If True, returns the total operator for the CZ gate. 
                If False, applies CZ gate operation to the quantum state. Defaults to False.

        Returns:
            ndarray: The resulting quantum state after applying the CZ gate operation.

        If props is True, the function calculates the total operator for the CZ gate
        and returns it. If props is False, the function applies the CZ gate operation
        to the quantum state and returns the resulting state.

        """
        if props == True:
            # Calculate the total operator for the CZ gate
            phi_B = np.pi/2
            t = np.pi/4 / omega_B
            chain_H_int = H_int_chain(self.chain, self.S, self.V)
            chain_H_0 = H_0_chain(self.chain, 0, 0, 0, omega_B, phi_B, 0, self.S)
            chain_H = chain_H_0 + chain_H_int
            U1 = time_evolution(chain_H, t)

            phi_B = 0
            t = np.pi / omega_B
            chain_H_0 = H_0_chain(self.chain, 0, 0, 0, omega_B, phi_B, 0, self.S)
            chain_H = chain_H_0 + chain_H_int
            U2 = time_evolution(chain_H, t)

            phi_B = np.pi/2
            t = np.pi/2 / omega_B
            chain_H_0 = H_0_chain(self.chain, 0, 0, 0, omega_B, phi_B, 0, self.S)
            chain_H = chain_H_0 + chain_H_int
            U3 = time_evolution(chain_H, t)

            return U1 @ U2 @ U3 @ U2 @ U1  # Return the total operator for the CZ gate
            
        else:
            phi_B =  np.pi/2
            t = np.pi/4 / omega_B
            dt = t / self.n_slices
            chain_H_int = H_int_chain(self.chain, self.S, self.V)
            chain_H_0 = H_0_chain(self.chain, 0, 0, 0, omega_B, phi_B, 0, self.S)
            chain_H = chain_H_0 + chain_H_int
            U1 = time_evolution(chain_H, dt)

            phi_B = 0
            t = np.pi / omega_B
            dt = t / self.n_slices
            chain_H_0 = H_0_chain(self.chain, 0, 0, 0, omega_B, phi_B, 0, self.S)
            chain_H = chain_H_0 + chain_H_int
            U2 = time_evolution(chain_H, dt)
            
            phi_B = np.pi/2
            t = np.pi/2 / omega_B
            dt = t / self.n_slices
            chain_H_0 = H_0_chain(self.chain, 0, 0, 0, omega_B, phi_B, 0, self.S)
            chain_H = chain_H_0 + chain_H_int
            U3 = time_evolution(chain_H, dt)

            for i in range(self.n_slices):
                self.psi = U1 @ self.psi
                self.psi_t.append(self.psi)

            for i in range(self.n_slices):
                self.psi = U2 @ self.psi
                self.psi_t.append(self.psi)
            
            for i in range(self.n_slices):
                self.psi = U3 @ self.psi
                self.psi_t.append(self.psi)

            for i in range(self.n_slices):
                self.psi = U2 @ self.psi
                self.psi_t.append(self.psi)

            for i in range(self.n_slices):
                self.psi = U1 @ self.psi
                self.psi_t.append(self.psi)

            self.N_op += 5
            return self.psi


    def Hadamard_S2(self, omega_A, omega_B, phi_B, props=False, N=2, params=[2.4115961, 3.14159335, 2.98995701, 3.14159277]):
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
        # Calculate interaction Hamiltonian
        chain_H_int = H_int_chain(self.chain, self.S, self.V)
        U_i = []

        if props:
            # Apply N concatenated Hadamard gates
            for i in range(N):
                phi_A, t = params[i*2], (params[i*2+1] / omega_A)
                
                # Calculate U for each gate
                chain_H_0 = H_0_chain(self.chain, omega_A, phi_A, 0, 0, 0, 0, self.S)
                chain_H = chain_H_0 + chain_H_int
                Ui = time_evolution(chain_H, t)
                
                if i == 0:
                    U = Ui
                else:
                    U = Ui @ U

            # Calculate the adjoint of U
            U_dag = U.conj().T
            Ztot = self.Z_tot(omega_B, phi_B, props=True)
            return U_dag @ Ztot @ U

        else:
            # Apply N concatenated Hadamard gates
            for i in range(N):
                phi_A, t = params[i*2], (params[i*2+1] / omega_A)
                dt = t / 100
                for ii in range(100):
                    chain_H_0 = H_0_chain(self.chain, omega_A, phi_A, 0, 0, 0, 0, self.S)
                    chain_H = chain_H_0 + chain_H_int
                    Ui = time_evolution(chain_H, dt)
                    U_i.append(Ui)
                    self.psi = Ui @ self.psi
                    self.psi_t.append(self.psi)
            self.N_op += 2
            _ = self.Z_tot(omega_B, phi_B)
            # Apply the adjoint of U for each dt
            for i in range(len(U_i)-1, -1, -1):
                # Apply the adjoint of U
                self.psi = U_i[i].conj().T @ self.psi
                self.psi_t.append(self.psi)
            self.N_op += 2

            return self.psi


    def CZ_S2(self, omega_B, phi_B, props=False, params=None):
        """
        Implements the controlled-Z (CZ) gate operation of the paper arXiv:2305.19220v2 adapted to S=2.

        Parameters:
            omega_B (float): Rabi frequency of the control qubit.
            phi_B (float): Phase of the control qubit.
            props (bool): If True, returns the total operator for the CZ gate; 
                        if False, applies the CZ gate operation to the quantum state.
            params (array-like, optional): Parameters for the CZ gate operation.

        Returns:
            ndarray or None: If props is True, returns the total operator for the CZ gate;
                            if props is False, returns None.
        """
        if params is None:
            params = np.array([np.pi/2, np.pi/4, 0, np.pi, np.pi/2, np.pi/2, 0, np.pi, np.pi/2, np.pi/4])
            params[1::2] *= 2 / np.sqrt(self.S)  # Scale times for different superatom values

        chain_H_int = H_int_chain(self.chain, self.S, self.V)
        U_i = []
        N = len(params) // 2  # Number of concatenated pulses

        if props:
            for i in range(N):
                phi_B, t = params[i*2], (params[i*2+1] / omega_B)
                # Calculation of U   
                chain_H_0 = H_0_chain(self.chain, 0, 0, 0, omega_B, phi_B, 0, self.S)
                chain_H = chain_H_0 + chain_H_int
                Ui = time_evolution(chain_H, t)
                if i == 0:
                    U = Ui
                else:
                    U = Ui @ U

            return U

        else:
            for i in range(N):
                phi_B, t = params[i*2], (params[i*2+1] / omega_B)
                # Calculation of U
                dt = t / 100
                for ii in range(100):
                    chain_H_0 = H_0_chain(self.chain, 0, 0, 0, omega_B, phi_B, 0, self.S)
                    chain_H = chain_H_0 + chain_H_int
                    Ui = time_evolution(chain_H, dt)
                    U_i.append(Ui)
                    self.psi = Ui @ self.psi
                    self.psi_t.append(self.psi)
            self.N_op += 5
            return self.psi


    def CZ_qoc(self, omega_B, phi_B, props=False, pulse=None, timegrid0=None):
        """
        Implements the controlled-Z (CZ) gate operation using quantum optimal control: dCRAB with symmetric pulses.

        Parameters:
            omega_B (float): Rabi frequency of the control qubit.
            phi_B (float): Phase of the control qubit.
            props (bool): If True, returns the total operator for the CZ gate;
                        if False, applies the CZ gate operation to the quantum state.
            pulse (array-like, optional): Custom pulse shape for the control qubit.
            timegrid0 (array-like, optional): Custom time grid for the pulse.

        Returns:
            ndarray or None: If props is True, returns the total operator for the CZ gate;
                            if props is False, returns None.
        """
        if pulse is None:
            pulse0 = list(np.load("pulse0.npy"))
            timegrid0 = list(np.load("timegrid0.npy"))
        else:
            pulse0 = pulse
            timegrid0 = timegrid0
        
        n_slices = 100
        H_control = [get_control_hamiltonian_ii(self.chain, self.S, 1).toarray()]
        H_drift = get_static_hamiltonian(self.chain, self.S, self.V, 0, 0, 0, omega_B, 0, 0).toarray()
        pulses = [pulse0]
        timegrid = [timegrid0]
        dt = timegrid0[1] - timegrid0[0]
        props_0 = [np.eye(2**self.N, dtype=np.complex128) for i in range(n_slices)]
        U = time_evolution_qoc(props_0, pulses, H_drift, H_control, 100, dt)
        
        if props:
            U_1 = reduce(lambda a, b: a @ b, U)
        else: 
            for i in range(n_slices):
                self.psi = U[n_slices-i-1] @ self.psi
                self.psi_t.append(self.psi)

        flipped_pulses = [np.flip(pulse0)]
        props_0 = [np.eye(2**self.N, dtype=np.complex128) for i in range(n_slices)]
        U = time_evolution_qoc(props_0, flipped_pulses, H_drift, H_control, 100, dt)
        
        if props:
            U_2 = reduce(lambda a, b: a @ b, U)
        else:
            for i in range(n_slices):
                self.psi = U[n_slices-i-1] @ self.psi
                self.psi_t.append(self.psi)
            
        if props:
            return U_2 @ U_1
        else:
            self.N_op += 2
            return self.psi
        
    def Bloch_sphere_plot(self, z_labels, indexes_0, indexes_1, angles=[90, 15], filename=None):
        """
        Print the evolution of the state in a Bloch sphere.

        Parameters:
            z_labels: list - List of labels for the z-axis.
            indexes_0: list - List of indices for the first basis vector.
            indexes_1: list - List of indices for the second basis vector.
            angles: list, optional - Viewing angles for the Bloch sphere. Defaults to [90, 15].
        """
        b = Bloch(view=angles)
        b.point_marker = ['o']
        colors = ['r']
        colors += ['b' for i in range(self.n_slices*self.N_op - 5)]
        for i in range(5):
            colors.append('g')
        b.point_color = colors
        b.point_size = [15]
        b.font_size = 18
        b.zlpos = [1.2, -1.2]
        b.zlabel = z_labels
        for psi_i in self.psi_t:
            c_0 = np.sum(psi_i[indexes_0])/np.sqrt(len(indexes_0))
            c_1 = np.sum(psi_i[indexes_1])/np.sqrt(len(indexes_1))
            vec = (c_0 * basis(2, 0) + c_1 * basis(2, 1))
            b.add_states(vec, kind='point')

        b.figsize = [5, 10]
        b.render()
        # save with the filename
        if filename is not None:
            b.save(filename)
            print("Bloch sphere plot saved as", filename)
        else:
            b.save()
