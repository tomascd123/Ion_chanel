import MDAnalysis as mda
from joblib import  Parallel, delayed
from MDAnalysis.analysis import  distances
from matplotlib.pyplot import close

topology = 'step5_input.psf'
trajectory_files = [
    'step6.1_equilibration.dcd',
    'step6.2_equilibration.dcd',
    'step6.3_equilibration.dcd',
    'step6.4_equilibration.dcd',
    'step6.5_equilibration.dcd',
    'step6.6_equilibration.dcd',
    'step7_production.dcd',
    'step8_production.dcd'
]
u = mda.Universe(topology, trajectory_files)
# Selección del residuo fijo
resid_3 = u.select_atoms('resid 146 and name Cl-')

# Definir distancia mínima (en angstroms)
distancia_minima = 5.0


# Función para procesar un frame individual
def procesar_frame(ts):
    # Seleccionar átomos que están dentro de cierta distancia del residuo de interés
    vecinos = u.select_atoms(f'around {distancia_minima} resid {resid_3.resids[0]}')
    # Retornar los residuos cercanos en este frame
    return [res.resid for res in vecinos.residues]


# Usar joblib para paralelización de frames
cercanos_residuos = Parallel(n_jobs=-1)(delayed(procesar_frame)(ts) for ts in u.trajectory)

# Aplanar la lista de resultados
cercanos_residuos = [residuo for sublist in cercanos_residuos for residuo in sublist]

# Opcional: filtrar y contar residuos únicos
cercanos_residuos_unicos = set(cercanos_residuos)
print(f"Residuos cercanos únicos que rodean a resid_3: {cercanos_residuos_unicos}")
