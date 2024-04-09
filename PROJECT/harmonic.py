import numpy as np
from numpy import linalg as la
import numpy as np
from numpy import linalg as la

def compute_matter_matrix_element(bra_nm, bra_np, ket_nm, ket_np, k, mu):
    """ Applies to all three Hamiltonians """
    return np.sqrt( k / mu) * (ket_nm + 1/2) * (bra_nm == ket_nm) * (bra_np == ket_np)
       
    
def compute_photon_matrix_element(bra_nm, bra_np, ket_nm, ket_np, omega_p):
    """ Applies to all three Hamiltonians """
    return omega_p * (ket_np + 1/2) * (bra_nm == ket_nm) * (bra_np == ket_np)
    

def compute_diamagnetic_element_p_dot_A(bra_nm, bra_np, ket_nm, ket_np, z_charge, A0, mu):
    """
    z ** 2 / 2m * A0 ** 2 * (b^+ + b)^2
    """
    fac = z_charge ** 2 * A0 ** 2 / ( 2 * mu)

    # must be diagonal in photon space
    val = 0
    if bra_nm == ket_nm:
        if bra_np == ket_np:
            val = 2 * fac * (ket_np + 1/2)
        
        elif bra_np == ket_np + 2:
            val = fac * np.sqrt(ket_np + 1) * np.sqrt(ket_np + 2)
        elif bra_np == ket_np - 2:
            val = fac * np.sqrt(ket_np) * np.sqrt(ket_np - 1)
        else:
            val = 0
            
    return val
    

    
    
def compute_interaction_matrix_element_p_dot_A(bra_nm, bra_np, ket_nm, ket_np, omega_p, z_charge, A0, k, mu):
    hbar = 1 # plancks constant / 2 * pi in atomic units
    omega_m = np.sqrt( k / mu) 
    p0 = 1j * np.sqrt(mu * hbar * omega_m / 2)
    
    fac = -z_charge * A0 * p0 / mu
    #print(F'pda fact is {fac}')
    
    # matter terms
    if bra_nm == ket_nm+1:
        term_1 = np.sqrt(ket_nm + 1)
    else:
        term_1 = 0
     
    if bra_nm == ket_nm-1:
        term_2 = -np.sqrt(ket_nm)
    else:
        term_2 = 0
    
    # photon terms
    if bra_np == ket_np+1:
        term_3 = np.sqrt(ket_np + 1)
    else:
        term_3 = 0
        
    if bra_np == ket_np-1:
        term_4 = np.sqrt(ket_np)
    else:
        term_4 = 0
        
    return fac * (term_1 + term_2) * (term_3 + term_4)

def compute_interaction_matrix_element_PF(bra_nm, bra_np, ket_nm, ket_np, omega_p, z_charge, A0, k, mu):
    """
     - \omega \hat{\mu} \cdot {\bf A}_0 ( \hat{b}^{\dagger} + \hat{b})
    """
    
    hbar = 1 # plancks constant / 2 * pi in atomic units
    omega_m = np.sqrt( k / mu) 
    x0 = np.sqrt( 1 / (2 * mu * omega_m))
    
    fac = -omega_p * z_charge * x0 * A0
    
    # matter terms
    if bra_nm == ket_nm+1:
        term_1 = np.sqrt(ket_nm + 1)
    else:
        term_1 = 0
     
    if bra_nm == ket_nm-1:
        term_2 = np.sqrt(ket_nm)
    else:
        term_2 = 0
    
    # photon terms
    if bra_np == ket_np+1:
        term_3 = np.sqrt(ket_np + 1)
    else:
        term_3 = 0
        
    if bra_np == ket_np-1:
        term_4 = np.sqrt(ket_np)
    else:
        term_4 = 0
        
    return fac * (term_1 + term_2) * (term_3 + term_4)



def compute_dipole_self_energy_PF(bra_nm, bra_np, ket_nm, ket_np, omega_p, z_charge, A0, k, mu):
    """
    +frac{\omega_{{\rm cav}}}{\hbar} ( \hat{\mu} \cdot {\bf A}_0)^2
    """
    hbar = 1
    omega_m = np.sqrt( k / mu )
    x0 = np.sqrt( 1 / (2 * mu * omega_m) )
    fac = omega_p * z_charge ** 2 * x0 ** 2 * A0 ** 2

    # must be diagonal in photon space
    val = 0
    if bra_np == ket_np:
        if bra_nm == ket_nm:
            val = 2 * fac * (ket_nm + 1/2)
        
        elif bra_nm == ket_nm + 2:
            val = fac * np.sqrt(ket_nm + 1) * np.sqrt(ket_nm + 2)
        elif bra_nm == ket_nm - 2:
            val = fac * np.sqrt(ket_nm) * np.sqrt(ket_nm - 1)
        else:
            val = 0
            
    return val
    

def build_and_diagonalize_p_dot_A(basis, k, mu, omega, z, A0):
    # length of slice of first column gives us the dimension of the Hamiltonian
    dim = len(basis[:,0])
    
    # initialize our Hamiltonian
    H_pda = np.zeros((dim,dim), dtype=complex)


    ket_idx = 0
    for ket in basis:

        bra_idx = 0
        
        for bra in basis:
            # matter term
            H_m_element = compute_matter_matrix_element(bra[0], bra[1], ket[0], ket[1], k, mu)
            # photon term
            H_p_element = compute_photon_matrix_element(bra[0], bra[1], ket[0], ket[1], omega)
            # interaction term
            H_i_element = compute_interaction_matrix_element_p_dot_A(bra[0], bra[1], ket[0], ket[1], omega, z, A0, k, mu)
            # diamagnetic term 
            H_d_element = compute_diamagnetic_element_p_dot_A(bra[0], bra[1], ket[0], ket[1], z, A0, mu)

            H_pda[bra_idx, ket_idx] = H_m_element + H_p_element + H_i_element + H_d_element
            bra_idx = bra_idx + 1
        ket_idx = ket_idx + 1 #ket_idx += 1
    
    # compute eigenvalues and eigenvectors
    vals, vecs = la.eigh(H_pda)
    
    # only return vals
    return H_pda, vals



def build_and_diagonalize_PF(basis, k, mu, omega, z, A0):
    # length of slice of first column gives us the dimension of the Hamiltonian
    dim = len(basis[:,0])
    
    # initialize our Hamiltonian
    H_PF = np.zeros((dim,dim), dtype=complex)


    ket_idx = 0
    for ket in basis:

        bra_idx = 0
        
        for bra in basis:

            H_m_element = compute_matter_matrix_element(bra[0], bra[1], ket[0], ket[1], k, mu)

            H_p_element = compute_photon_matrix_element(bra[0], bra[1], ket[0], ket[1], omega)

            H_i_element = compute_interaction_matrix_element_PF(bra[0], bra[1], ket[0], ket[1], omega, z, A0, k, mu)

            H_dse_element = compute_dipole_self_energy_PF(bra[0], bra[1], ket[0], ket[1], omega, z, A0, k, mu)

            H_PF[bra_idx, ket_idx] = H_m_element + H_p_element + H_i_element + H_dse_element
            bra_idx = bra_idx + 1
        ket_idx = ket_idx + 1 #ket_idx += 1
    
    # compute eigenvalues and eigenvectors
    vals, vecs = la.eigh(H_PF)
    
    # only return vals
    return H_PF, vals
