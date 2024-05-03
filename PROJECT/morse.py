import numpy as np
from scipy.constants import h, hbar, c, u
from scipy.special import factorial
from scipy.special import genlaguerre, gamma

# Factor for conversion from cm-1 to J
FAC = 100 * h * c

class Morse:
    """A class representing the Morse oscillator model of a diatomic."""

    def __init__(self, mA, mB, we, wexe, re, Te=0, dipole=0, A0=0, omega_p = 0, matter_dim=2, photon_dim=2):
        """Initialize the Morse model for a diatomic molecule.

        mA, mB are the atom masses (atomic mass units).
        we, wexe are the Morse parameters (cm-1).
        re is the equilibrium bond length (m).
        Te is the electronic energy (minimum of the potential well; origin
            of the vibrational state energies).
        dipole is the total dipole moment in atomic units

        """
        # size of photon and matter basis
        self.matter_dim = matter_dim
        self.photon_dim = photon_dim

        # magnitude of the vector potential
        self.A0_au = A0
        # dipole moment
        self.dipole = dipole 
        # atomic mass units to kg
        self.amu_to_kg = 1.66054e-27
        
        # angstroms to meters
        self.ang_to_m = 1e-10
        
        # electron volts to Jouls
        self.eV_to_J = 1.60218e-19
        
        # electron volts to atomic units of energy (Hartrees)
        self.eV_to_au = 1 / 27.211 #0.0367493
        
        # angstroms to atomic units of length (Bohr radii)
        self.au_to_ang = 0.52917721067121

        # meters to atomic units
        self.m_to_au = 1 / self.ang_to_m * 1 / self.au_to_ang

        # atomic units to wavenumbers
        self.au_to_wn = 219474.63068
        
        # atomic mass units to atomic units of mass
        self.amu_to_au = 1822.89

        # masses originally in atomic mass units
        self.mA, self.mB = mA, mB

        # reduced mass in SI units
        self.mu = mA*mB/(mA+mB) * u

        # reduced mass in atomic units
        self.mu_au = mA * mB/(mA + mB) * self.amu_to_au

        #  Morse parameters in wavenumbers
        self.we, self.wexe = we, wexe

        # Morse parameters in atomic units
        self.we_au = self.we / self.au_to_wn
        self.wexe_au = self.wexe / self.au_to_wn

        # equilibrium bondlength in SI units
        self.re = re

        # energy offset in atomic units
        self.Te = Te

        # dissociation energy in Joules
        self.De = we**2 / 4 / wexe * FAC

        # force constant 
        self.ke = (2 * np.pi * c * 100 * we)**2 * self.mu

        #  Morse parameters, a and lambda.
        self.a = self.calc_a()
        self.lam = np.sqrt(2 * self.mu * self.De) / self.a / hbar

        #self.De_au = 

        # Maximum vibrational quantum number.
        self.vmax = int(np.floor(self.lam - 0.5))

        # grid in SI
        self.make_rgrid()

        # potential in SI
        self.V = self.Vmorse(self.r)

        # potential in atomic units
        self.V_au = self.V / self.eV_to_J * self.eV_to_au

        # grid in atomic units
        self.r_au = self.r * self.m_to_au

        # equilibrium bondlength in atomic units
        self.r_eq_au = re * self.m_to_au

        # charge in atomic units
        self.q_au = self.dipole / self.r_eq_au

        # parameter z in SI
        self.z = 2 * self.lam * np.exp(-self.a * (self.r - self.re))

        # photon energy

        if omega_p == 0:
            omega_f = self.compute_Morse_transition_au(0, 1)
            self.omega_p = omega_f
        else:
            self.omega_p = omega_p
        
    

    def make_rgrid(self, n=500, rmin=None, rmax=None, retstep=False):
        """Make a suitable grid of internuclear separations."""

        self.rmin, self.rmax = rmin, rmax
        if rmin is None:
            # minimum r where V(r)=De on repulsive edge
            self.rmin = self.re - 1.75 * np.log(2) / self.a
        if rmax is None:
            # maximum r where V(r)=f.De
            f = 0.999
            self.rmax = self.re - 1.25 *  np.log(1-f)/self.a
        self.r, self.dr = np.linspace(self.rmin, self.rmax, n,
                                      retstep=True)
        if retstep:
            return self.r, self.dr
        return self.r

    def calc_a(self):
        """Calculate the Morse parameter, a.

        Returns the Morse parameter, a, from the equilibrium
        vibrational wavenumber, we in cm-1, and the dissociation
        energy, De in J.

        """

        return (self.we * np.sqrt(2 * self.mu/self.De) * np.pi *
                c * 100)

    def Vmorse(self, r):
        """Calculate the Morse potential, V(r).

        Returns the Morse potential at r (in m) for parameters De
        (in J), a (in m-1) and re (in m).

        """

        return self.De * (1 - np.exp(-self.a*(r - self.re)))**2

    def Emorse(self, v):
        """Calculate the energy of a Morse oscillator in state v.

        Returns the energy of a Morse oscillator parameterized by
        equilibrium vibrational frequency we and anharmonicity
        constant, wexe (both in cm-1).

        """
        vphalf = v + 0.5
        return (self.we * vphalf - self.wexe * vphalf**2) * FAC

    def calc_turning_pts(self, E):
        """Calculate the classical turning points at energy E.

        Returns rm and rp, the classical turning points of the Morse
        oscillator at energy E (provided in J). rm < rp.

        """

        b = np.sqrt(E / self.De)
        return (self.re - np.log(1+b) / self.a,
                self.re - np.log(1-b) / self.a)

    def calc_psi(self, v, r=None, normed=True, psi_max=1):
        """Calculates the Morse oscillator wavefunction, psi_v.

        Returns the Morse oscillator wavefunction at vibrational
        quantum number v. The returned function is "normalized" to
        give peak value psi_max.

        """

        if r is None:
            r = self.r
        z = 2 * self.lam * np.exp(-self.a*(r - self.re))
        alpha = 2*(self.lam - v) - 1
        psi = (z**(self.lam-v-0.5) * np.exp(-z/2) *
               genlaguerre(v, alpha)(z))
        rho = psi * np.conj(psi)
	#psi *= np.conj(psi)
        psi *= psi_max / np.sqrt(np.max(rho))
        return psi

    def calc_psi_z(self, v):
        z = self.z 
        alpha = 2*(self.lam - v) - 1
        psi = (z**(self.lam-v-0.5) * np.exp(-z/2) *
               genlaguerre(v, alpha)(z))
        Nv = np.sqrt(factorial(v) * (2*self.lam - 2*v - 1) /
                     gamma(2*self.lam - v))
        
        self.psi_si = psi * Nv
        self.norm_au = np.trapz(self.psi_si ** 2, self.r_au)
        self.psi_au = self.psi_si / np.sqrt(self.norm_au)
        self.norm_si = np.trapz(self.psi_si ** 2, self.r)
        self.psi_si /= np.sqrt(self.norm_si)
        return Nv * psi

    def plot_V(self, ax, **kwargs):
        """Plot the Morse potential on Axes ax."""

        ax.plot(self.r*1.e10, self.V / FAC + self.Te, **kwargs)

    def get_vmax(self):
        """Return the maximum vibrational quantum number."""

        return int(self.we / 2 / self.wexe - 0.5)

    def draw_Elines(self, vlist, ax, **kwargs):
        """Draw lines on Axes ax representing the energy level(s) in vlist."""

        if isinstance(vlist, int):
            vlist = [vlist]
        for v in vlist:
            E = self.Emorse(v)
            rm, rp = self.calc_turning_pts(E)
            ax.hlines(E / FAC + self.Te, rm*1.e10, rp*1e10, **kwargs)

    def label_levels(self, vlist, ax):
        if isinstance(vlist, int):
            vlist = [vlist]

        for v in vlist:
            E = self.Emorse(v)
            rm, rp = self.calc_turning_pts(E)
            ax.text(s=r'$v={}$'.format(v), x=rp*1e10 + 0.6,
                    y=E / FAC + self.Te, va='center')

    def plot_psi(self, vlist, ax, r_plot=None, scaling=1, **kwargs):
        """Plot the Morse wavefunction(s) in vlist on Axes ax."""
        if isinstance(vlist, int):
            vlist = [vlist]
        for v in vlist:
            E = self.Emorse(v)
            if r_plot is None:
                rm, rp = self.calc_turning_pts(E)
                x = self.r[self.r<rp*1.2]
            else:
                x = r_plot
            psi = self.calc_psi(v, r=x, psi_max=self.we/2)
            psi_plot = psi*scaling + self.Emorse(v)/FAC + self.Te
            ax.plot(x*1.e10, psi_plot, **kwargs)

    def compute_coupling_element_p_dot_A(self, bra_m, bra_p, ket_m, ket_p):
        """ Function to compute the matrix elements  
        z / m * <m|<p| \hat{p} | p'>|m'> = i * z * (E_m - E_m') A_0 * <m|\hat{x}|m'> * <p|(b^+ + b)|p'>
        """
        # imaginary unit
        ci = 0+1j
        
        # charge
        z = self.q_au
        
        # magnitude of vector potential
        A0 = self.A0_au

        # (Em - Em')
        deltaE = self.compute_Morse_transition_au(ket_m, bra_m)

        # <m|\hat{x}|m'>
        x_val = self.position_matrix_element(bra_m, ket_m)

        coupling_val = ci * z * deltaE * A0 * x_val * (np.sqrt(ket_p + 1 ) * (bra_p == ket_p + 1) + np.sqrt(ket_p) * (bra_p == ket_p - 1))
        return coupling_val
    
    def compute_coupling_element_d_dot_E(self, bra_m, bra_p, ket_m, ket_p):
        """ Function to compute the matrix elements  
        i \omega z <m|<p| \hat{x} \hat{A} | p'>|m'> = i * omega * z * A_0 <m|\hat{x}|m'> * <p|(b^+ - b)|p'>
        """
        # imaginary unit
        ci = 0+1j
        
        # charge
        z = self.q_au

        # omega
        omega = self.omega_p
        
        # magnitude of vector potential
        A0 = self.A0_au

        # <m|\hat{x}|m'>
        x_val = self.position_matrix_element(bra_m, ket_m)

        coupling_val = ci * z * omega * A0 * x_val * (np.sqrt(ket_p + 1 ) * (bra_p == ket_p + 1) - np.sqrt(ket_p) * (bra_p == ket_p - 1))
        return coupling_val
    
    def compute_coupling_element_PF(self, bra_m, bra_p, ket_m, ket_p):
        """ Function to compute the matrix elements  
        - omega * z <m|<p| \hat{x} \hat{A} | p'>|m'> = - omega * z * A_0 <m|\hat{x}|m'> * <p|(b^+ + b)|p'>
        """        
        # charge
        z = self.q_au

        # omega
        omega = self.omega_p
        
        # magnitude of vector potential
        A0 = self.A0_au

        # <m|\hat{x}|m'>
        x_val = self.position_matrix_element(bra_m, ket_m)

        coupling_val = -1 * z * omega * A0 * x_val * (np.sqrt(ket_p + 1 ) * (bra_p == ket_p + 1) + np.sqrt(ket_p) * (bra_p == ket_p - 1))
        return coupling_val
    

    def compute_dipole_self_energy_element(self, bra_m, bra_p, ket_m, ket_p):
        """ Function to compute the matrix elements
            
        """
        # must be diagonal
        val = 0
        if bra_p == ket_p:
            # collect terms
            omega = self.omega_p
            z = self.q_au
            x2_val = self.position_squared_matrix_element(bra_m, ket_m)
            A0 = self.A0_au


            # compute the matrix element
            val = omega * z ** 2 * A0 ** 2 * x2_val

        return val
    
    def compute_photon_element(self, bra_m, bra_p, ket_m, ket_p):
        """ Function to compute the matrix elements
            (omega)  * <m|<p|  (b^+ b + 1/2) | p'>|m'>
        """
        # must be diagonal
        val = 0
        if bra_m == ket_m and bra_p == ket_p:
            # collect terms
            omega = self.omega_p

            # compute the matrix element
            val = omega * (ket_p + 1/2)

        return val
    

    def compute_diamagnetic_element(self, bra_m, bra_p, ket_m, ket_p):
        """ Function to compute the matrix elements
            z^2 / 2m * A_0 <m|m'><p | (b^+ b^+ + bb + 2 b^+ b + 1) |p'>
        """
        # must be diagonal in matter states
        z = self.q_au
        A0 = self.A0_au
        mu = self.mu_au

        fac = z ** 2 * A0 ** 2 / (2 * mu)

        val = 0
        if bra_m == ket_m:
            if bra_p == ket_p:
                val = 2 * fac * (ket_p + 1/2)

            elif bra_p == ket_p + 2:
                val = fac * np.sqrt(ket_p + 1) * np.sqrt(ket_p + 2)

            elif bra_p == ket_p - 2:
                val = fac * np.sqrt(ket_p - 1) * np.sqrt(ket_p - 2)

            else:
                val = 0

        return val

    
    def compute_matter_element(self, bra_m, bra_p, ket_m, ket_p):
        """ Function to compute the matrix elements
            <m| p^2/2m + V(x) |m'><p|p'>
        """
        # must be diagonal in all states
        val = 0
        if bra_m == ket_m and bra_p == ket_p:
            val = self.we_au * (ket_m + 1/2) - self.wexe_au * (ket_m + 1/2) ** 2

        return val
    

    def build_basis(self):
        self.basis = []
        for i in range(self.photon_dim):
            for j in range(self.matter_dim):
                self.basis.append((j,i))

    def build_p_dot_A_Hamiltonian(self):
        # build the basis
        self.build_basis()
        dim = len(self.basis)

        self.H_p_dot_A = np.zeros((dim,dim), dtype=complex)
        for i in range(dim):
            bra_m = self.basis[i][0]
            bra_p = self.basis[i][1]
            for j in range(dim):
                ket_m = self.basis[j][0]
                ket_p = self.basis[j][1]

                H_matter = self.compute_matter_element(bra_m, bra_p, ket_m, ket_p)
                H_diam   = self.compute_diamagnetic_element(bra_m, bra_p, ket_m, ket_p)
                H_pho    = self.compute_photon_element(bra_m, bra_p, ket_m, ket_p)
                H_coup   = self.compute_coupling_element_p_dot_A(bra_m, bra_p, ket_m, ket_p)
                self.H_p_dot_A[i,j] = H_matter + H_diam + H_pho + H_coup

    def build_d_dot_E_Hamiltonian(self):
        # build the basis
        self.build_basis()
        dim = len(self.basis)

        self.H_d_dot_E = np.zeros((dim,dim), dtype=complex)
        for i in range(dim):
            bra_m = self.basis[i][0]
            bra_p = self.basis[i][1]
            for j in range(dim):
                ket_m = self.basis[j][0]
                ket_p = self.basis[j][1]

                H_matter = self.compute_matter_element(bra_m, bra_p, ket_m, ket_p)
                H_dse   = self.compute_dipole_self_energy_element(bra_m, bra_p, ket_m, ket_p)
                H_pho    = self.compute_photon_element(bra_m, bra_p, ket_m, ket_p)
                H_coup   = self.compute_coupling_element_d_dot_E(bra_m, bra_p, ket_m, ket_p)
                self.H_d_dot_E[i,j] = H_matter + H_dse + H_pho + H_coup

    def build_PF_Hamiltonian(self):
        # build the basis
        self.build_basis()
        dim = len(self.basis)

        self.H_PF = np.zeros((dim,dim))
        self.H_matter = np.zeros((dim,dim))
        self.H_pho = np.zeros((dim,dim))
        self.H_coup = np.zeros((dim,dim))
        self.H_dse = np.zeros((dim,dim))
        for i in range(dim):
            bra_m = self.basis[i][0]
            bra_p = self.basis[i][1]
            for j in range(dim):
                ket_m = self.basis[j][0]
                ket_p = self.basis[j][1]

                self.H_matter[i,j] = self.compute_matter_element(bra_m, bra_p, ket_m, ket_p)
                self.H_dse[i,j]   = self.compute_dipole_self_energy_element(bra_m, bra_p, ket_m, ket_p)
                self.H_pho [i,j]   = self.compute_photon_element(bra_m, bra_p, ket_m, ket_p)
                self.H_coup[i,j]   = self.compute_coupling_element_PF(bra_m, bra_p, ket_m, ket_p)
                self.H_PF[i,j] = self.H_matter[i,j] + self.H_dse[i,j] + self.H_pho[i,j] + self.H_coup[i,j]

    def position_matrix_element(self, i, j):
        """ A function to compute position matrix elements between states i and j using grid x

        Arguments
        ---------
        instance : class instance
            the instance of the class you want to use for the states
            
        i : int
            index of bra state

        j : int
            index of ket state

        Returns
        -------
        x_ij : float
            the matrix element <i | x | j>

        """
        self.calc_psi_z(i)
        psi_i = self.psi_au
        self.calc_psi_z(j)
        psi_j = self.psi_au
        # note that if you displace by r_eq, the dipole expectation
        # value will be the fluctuation from the "harmonic" value
        # but this doesn't seem to impact Rabi splitting or absolute energies!
        x_hat = self.r_au # - self.r_eq_au 
        integrand = psi_i * x_hat * psi_j
        x_ij = np.trapz(integrand, x_hat)
        return x_ij
    
    def position_squared_matrix_element(self, i, j):
        """ A function to compute position squared matrix elements between states i and j using grid x

        Arguments
        ---------
        instance : class instance
            the instance of the class you want to use for the states
            
        i : int
            index of bra state

        j : int
            index of ket state

        Returns
        -------
        x_ij : float
            the matrix element <i | x^2 | j>

        """
        self.calc_psi_z(i)
        psi_i = self.psi_au
        self.calc_psi_z(j)
        psi_j = self.psi_au
        # note that if you displace by r_eq, the dipole expectation
        # value will be the fluctuation from the "harmonic" value
        # but this doesn't seem to impact Rabi splitting or absolute energies!
        x_hat = self.r_au # - self.r_eq_au
        integrand = psi_i * x_hat * x_hat * psi_j
        x_ij = np.trapz(integrand, x_hat)
        return x_ij

    def compute_Morse_transition_au(self, i, f):
        Ei = self.we_au * (i + 1/2) - self.wexe_au * (i + 1/2) ** 2 
        Ef = self.we_au * (f + 1/2) - self.wexe_au * (f + 1/2) ** 2 
        deltaE = Ef - Ei
        return deltaE

    def compute_Harmonic_transition_au(self, i, f):
        Ei = self.we_au * (i + 1/2)
        Ef = self.we_au * (f + 1/2) 
        deltaE = Ef - Ei
        return deltaE
    
    def compute_Morse_transition_wn(self, i, f):
        Ei = self.we * (i + 1/2) - self.wexe * (i + 1/2) ** 2 
        Ef = self.we * (f + 1/2) - self.wexe * (f + 1/2) ** 2 
        deltaE = Ef - Ei
        return deltaE

    def compute_Harmonic_transition_wn(self, i, f):
        Ei = self.we * (i + 1/2)
        Ef = self.we * (f + 1/2) 
        deltaE = Ef - Ei
        return deltaE
