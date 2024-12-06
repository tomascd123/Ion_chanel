import MDAnalysis as mda
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
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

resid_1 = u.select_atoms('resid 608 and name N1')
resid_2 = u.select_atoms('resid 607 and name N1')
resid_3 = u.select_atoms('resid 146 and name Cl-')

# Ensure selections return one atom each
assert len(resid_1) == 1, "resid_1 selection does not return exactly one atom"
assert len(resid_2) == 1, "resid_2 selection does not return exactly one atom"
assert len(resid_3) == 1, "resid_3 selection does not return exactly one atom"

distances_1 = []
distances_2 = []

for ts in u.trajectory:
    # Calculate distance from resid_3 to resid_1
    distance_1 = np.linalg.norm(resid_3.positions - resid_1.positions)
    distances_1.append(distance_1)
    # Calculate distance from resid_3 to resid_2
    distance_2 = np.linalg.norm(resid_3.positions - resid_2.positions)
    distances_2.append(distance_2)

distances_1 = np.array(distances_1)
distances_2 = np.array(distances_2)

# Filtrar distancias menores a 12 Å
filtered_distances_1 = [(i, d) for i, d in enumerate(distances_1) if d < 12]
filtered_distances_2 = [(i, d) for i, d in enumerate(distances_2) if d < 12]

# Separar frames y distancias después de filtrar
frames_1, distances_1_filtered = zip(*filtered_distances_1) if filtered_distances_1 else ([], [])
frames_2, distances_2_filtered = zip(*filtered_distances_2) if filtered_distances_2 else ([], [])

# Calcular los datos de densidad para las líneas de KDE
kde_1 = gaussian_kde(distances_1_filtered)
kde_x1 = np.linspace(min(distances_1_filtered), max(distances_1_filtered), 100)
kde_y1 = kde_1(kde_x1)

kde_2 = gaussian_kde(distances_2_filtered)
kde_x2 = np.linspace(min(distances_2_filtered), max(distances_2_filtered), 100)
kde_y2 = kde_2(kde_x2)

# Crear subplots: 2 filas, 2 columnas
fig = make_subplots(rows=2, cols=2,
                    subplot_titles=("Distancias a lo largo de la simulación - resid_1",
                                    "Distribución Vertical de Distancias < 12 Å - resid_1",
                                    "Distancias a lo largo de la simulación - resid_2",
                                    "Distribución Vertical de Distancias < 12 Å - resid_2"))

# Scatter plot para resid_1
fig.add_trace(go.Scatter(
    x=frames_1, y=distances_1_filtered, mode='markers', marker=dict(color='blue'),
    name='Distancia a resid_1 (< 12 Å)'
), row=1, col=1)

# Gráfico de violín para la densidad de resid_1
fig.add_trace(go.Violin(
    x=distances_1_filtered, line_color='blue', showlegend=False, orientation='v',
    box_visible=True, meanline_visible=True
), row=1, col=2)

# Línea de densidad para resid_1
fig.add_trace(go.Scatter(
    x=kde_x1, y=kde_y1, mode='lines', line=dict(color='blue'),
    name='Densidad de distancia a resid_1'
), row=1, col=2)

# Scatter plot para resid_2
fig.add_trace(go.Scatter(
    x=frames_2, y=distances_2_filtered, mode='markers', marker=dict(color='red'),
    name='Distancia a resid_2 (< 12 Å)'
), row=2, col=1)

# Gráfico de violín para la densidad de resid_2
fig.add_trace(go.Violin(
    x=distances_2_filtered, line_color='red', showlegend=False, orientation='v',
    box_visible=True, meanline_visible=True
), row=2, col=2)

# Línea de densidad para resid_2
fig.add_trace(go.Scatter(
    x=kde_x2, y=kde_y2, mode='lines', line=dict(color='red'),
    name='Densidad de distancia a resid_2'
), row=2, col=2)

# Actualizar el layout
fig.update_layout(height=800, width=1200, title_text="Visualización de Distancias con Plotly",
                  showlegend=True)

# Etiquetas de los ejes
fig.update_xaxes(title_text="Frame", row=1, col=1)
fig.update_yaxes(title_text="Distancia (Å)", row=1, col=1)
fig.update_xaxes(title_text="Densidad", row=1, col=2)
fig.update_yaxes(title_text="Distancia (Å)", row=2, col=1)
fig.update_xaxes(title_text="Frame", row=2, col=1)

# Mostrar gráfico
fig.show()

