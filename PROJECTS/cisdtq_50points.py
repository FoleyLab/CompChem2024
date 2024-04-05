import psi4
import numpy as np
mol_tmpl = """
H
F 1 **R**
symmetry c2v
"""

r_angstroms = np.linspace( 0.5, 3.3, 50)
print(r_angstroms)
options_dict = {
    "basis": "6-311++G**",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
    "num_frozen_docc" : 1
}

psi4.set_options(options_dict)

ccsd_energies = []
cisdtq_energies = []

for r_val in r_angstroms:
    mol_str = mol_tmpl.replace("**R**", str(r_val))
    mol = psi4.geometry(mol_str)
    e_ccsd = psi4.energy('ccsd')
    ccsd_energies.append(e_ccsd)
    e_cisdtq = psi4.energy('cisdtq')
    cisdtq_energies.append(e_cisdtq)


print(r_angstroms)
print(ccsd_energies)
print(cisdtq_energies)

