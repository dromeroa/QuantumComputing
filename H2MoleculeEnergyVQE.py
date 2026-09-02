# ============================================================
#  Energía del estado fundamental de H2 con Qiskit + VQE
#  Actualizado para Qiskit Nature 1.0x
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# --- Qiskit Nature (química cuántica) ---
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

# --- Qiskit Algorithms (VQE + optimizador) ---
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import StatevectorEstimator as Estimator

# ============================================================
# 1. DEFINIR LA MOLÉCULA
# ============================================================
bond_distance = 0.735

driver = PySCFDriver(
    atom=f"H 0 0 0; H 0 0 {bond_distance}",
    basis="sto-3g",
    charge=0,
    spin=0,
)

problem = driver.run()
print("Propiedades de la molécula:")
print(f"  Nro. de orbitales espaciales : {problem.num_spatial_orbitals}")
print(f"  Nro. de partículas           : {problem.num_particles}")



# ============================================================
# 2. MAPEO FERMIÓN → QUBIT (Jordan-Wigner)
# ============================================================
# Ya no se usa QubitConverter; pasamos el mapper directamente.
mapper = JordanWignerMapper()


# ============================================================
# 3. ANSATZ UCCSD + ESTADO INICIAL HARTREE-FOCK
# ============================================================
num_particles = problem.num_particles          
num_spatial_orbs = problem.num_spatial_orbitals  

# Estado inicial: Hartree-Fock (usando 'mapper' en lugar de 'converter')
hf_state = HartreeFock(
    num_spatial_orbs,
    num_particles,
    mapper,
)

# Ansatz variacional: UCCSD (usando 'mapper' en lugar de 'converter')
ansatz = UCCSD(
    num_spatial_orbs,
    num_particles,
    mapper,
    initial_state=hf_state,
)

print(f"\nCircuito UCCSD:")
print(f"  Nro. de qubits     : {ansatz.num_qubits}")
print(f"  Nro. de parámetros : {ansatz.num_parameters}")


# ============================================================
# 4. CONFIGURAR VQE
# ============================================================
optimizer = SLSQP(maxiter=300)

vqe_solver = VQE(
    estimator=Estimator(),
    ansatz=ansatz,
    optimizer=optimizer,
    initial_point=np.zeros(ansatz.num_parameters),
)


# ============================================================
# 5. RESOLVER EL ESTADO FUNDAMENTAL
# ============================================================
# GroundStateEigensolver ahora recibe el mapper directamente
solver = GroundStateEigensolver(mapper, vqe_solver)
result = solver.solve(problem)

print("\n" + "="*55)
print("  RESULTADO VQE — H2 (STO-3G)")
print("="*55)
print(f"  Energía total VQE        : {result.total_energies[0]:.8f} Ha")
print(f"  Energía HF de referencia : {problem.reference_energy:.8f} Ha")
print(f"  Correlación capturada    : "
      f"{result.total_energies[0] - problem.reference_energy:.6f} Ha")
print("="*55)


# ============================================================
# 6. CURVA DE ENERGÍA POTENCIAL 
# ============================================================
def energia_vs_distancia(distancias_ang):
    """Calcula E(R) para una lista de distancias de enlace."""
    energias = []
    for d in distancias_ang:
        drv = PySCFDriver(
            atom=f"H 0 0 0; H 0 0 {d}",
            basis="sto-3g", charge=0, spin=0,
        )
        prob = drv.run()
        ans  = UCCSD(prob.num_spatial_orbitals, prob.num_particles, mapper,
                     initial_state=HartreeFock(prob.num_spatial_orbitals,
                                               prob.num_particles, mapper))
        vqe  = VQE(Estimator(), ans, SLSQP(maxiter=300),
                   initial_point=np.zeros(ans.num_parameters))
        sol  = GroundStateEigensolver(mapper, vqe).solve(prob)
        energias.append(sol.total_energies[0])
        print(f"  d = {d:.3f} Å  →  E = {energias[-1]:.6f} Ha")
    return np.array(energias)

# Descomenta para trazar la curva completa:
distancias = np.linspace(0.5, 2.5, 20)
energias   = energia_vs_distancia(distancias)
plt.figure(figsize=(7, 4))
plt.plot(distancias, energias, 'o-', color='steelblue', lw=2)
plt.xlabel("Distancia H-H (Å)")
plt.ylabel("Energía (Hartree)")
plt.title("Curva de energía potencial — H₂ / VQE / STO-3G")
plt.axvline(0.735, ls='--', color='gray', label='Equilibrio 0.735 Å')
plt.legend(); plt.tight_layout(); plt.savefig("PEC_H2.pdf"); plt.show()
