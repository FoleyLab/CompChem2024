import numpy as np


def compute_matter_matrix_element(bra_nm, bra_np, ket_nm, ket_np, k, mu):
    hbar = 1 # this is atomic units!
    if bra_nm == ket_nm and bra_np == ket_np:
        return hbar * np.sqrt( k / mu) * (ket_nm + 1/2)
    
    else:
        return 0
    
    
def compute_photon_matrix_element(bra_nm, bra_np, ket_nm, ket_np, omega_p, z_charge, A0, mu):
    hbar = 1 # this is atomic units!
    if bra_nm == ket_nm and bra_np == ket_np:
        term_1 = hbar * omega_p * (ket_np + 1/2)
        term_2 = z_charge ** 2 * A0 ** 2 / mu * (ket_np + 1/2)
        return term_1 + term_2
    
    else: 
        return 0
    

def compute_photon_matrix_element_PF(bra_nm, bra_np, ket_nm, ket_np, omega_p, z_charge, A0, mu):
    hbar = 1 # this is atomic units!
    if bra_nm == ket_nm and bra_np == ket_np:
        term_1 = hbar * omega_p * (ket_np + 1/2)
        return term_1
    
    else: 
        return 0
    


def compute_interaction_matrix_element_PF(bra_nm, bra_np, ket_nm, ket_np, omega_p, z_charge, A0, k, mu):
    hbar = 1 # plancks constant / 2 * pi in atomic units
    omega_m = np.sqrt( k / mu) 
    p0 = 1j * np.sqrt(mu * hbar * omega_m / 2)
    
    fac = -z_charge * A0 * p0 / mu
    
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
        term_4 = np.sqrt(ket_nm)
    else:
        term_4 = 0
        
    return fac * (term_1 + term_2) * (term_3 + term_4)


    
def compute_interaction_matrix_element(bra_nm, bra_np, ket_nm, ket_np, omega_p, z_charge, A0, k, mu):
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
        term_4 = np.sqrt(ket_nm)
    else:
        term_4 = 0
        
    return fac * (term_1 + term_2) * (term_3 + term_4)

def build_and_diagonalize_p_dot_A(basis, k, mu, omega, z, A0):
    # length of slice of first column gives us the dimension of the Hamiltonian
    dim = len(basis[:,0])
    
    # initialize our Hamiltonian
    H_pda = np.zeros((dim,dim), dtype=complex)


    ket_idx = 0
    for ket in basis:

        bra_idx = 0
        
        for bra in basis:

            H_m_element = compute_matter_matrix_element(bra[0], bra[1], ket[0], ket[1], k, mu)

            H_p_element = compute_photon_matrix_element(bra[0], bra[1], ket[0], ket[1], omega, z, A0, mu)
 
            H_i_element = compute_interaction_matrix_element(bra[0], bra[1], ket[0], ket[1], omega, z, A0, k, mu)

            H_pda[bra_idx, ket_idx] = H_m_element + H_p_element + H_i_element
            bra_idx = bra_idx + 1
        ket_idx = ket_idx + 1 #ket_idx += 1
    
    # compute eigenvalues and eigenvectors
    vals, vecs = la.eigh(H_pda)
    
    # only return vals
    return vals


basis_array = np.array([[0,0], [1,0], [0,1], [1,1]])
k_val = 1
mu_val = 1
z_val = 1
omega_p_val = 1
A0_val = 0.

H_pda = np.zeros((4,4), dtype=complex)
print(H_pda)

ket_idx = 0
for ket in basis_array:
    print(F' matter basis state is |{ket[0]}> and photon basis state is |{ket[1]}>')
    bra_idx = 0
    for bra in basis_array:
        print(F' matter basis state is <{bra[0]}| and photon basis state is <{bra[1]}|')
        H_m_element = compute_matter_matrix_element(bra[0], bra[1], ket[0], ket[1], k_val, mu_val)
        print(H_m_element)
        H_p_element = compute_photon_matrix_element(bra[0], bra[1], ket[0], ket[1], omega_p_val, z_val, A0_val, mu_val)
        print(H_p_element)
        H_i_element = compute_interaction_matrix_element(bra[0], bra[1], ket[0], ket[1], omega_p_val, z_val, A0_val, k_val, mu_val)
        print(H_i_element)
        H_pda[bra_idx, ket_idx] = H_m_element + H_p_element + H_i_element
        bra_idx = bra_idx + 1
    ket_idx = ket_idx + 1 #ket_idx += 1
    
print(np.real(H_pda))
        
    
        
    
        
    
