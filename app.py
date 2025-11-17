# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import spectrogram
import time
import math
import pandas as pd

st.set_page_config(page_title="OceanSense Advanced", layout="wide", initial_sidebar_state="expanded")
st.title("🌊 OceanSense Advanced — Oceanography & Instrumentation Simulator")
st.write("All enhancements included: real-time buoy telemetry, 3D bathymetry, ROV/AUV sim, CTD profiles, pollution pathways, EEZ visualizer, noise dashboards, ecosystem & more — **no datasets, no models**.")

# -----------------------------
# Sidebar controls (global)
# -----------------------------
st.sidebar.header("Global Controls")
sim_speed = st.sidebar.selectbox("Simulation speed", ["Realtime", "Fast", "Paused"])
time_step = {"Realtime": 1.0, "Fast": 0.2, "Paused": None}[sim_speed]
seed = st.sidebar.number_input("Random seed", value=42, step=1)
np.random.seed(seed)

# -----------------------------
# Helper utilities
# -----------------------------
def sound_speed_unesco(T, S, z):
    # simplified UNESCO empirical formula
    return 1449.2 + 4.6*T - 0.055*(T**2) + 0.00029*(T**3) + (1.34 - 0.01*T)*(S-35) + 0.016*z

def density_approx(T, S):
    return 1000 + 0.8*S - 0.2*T

def perlin_noise(shape, scale=10, seed=0):
    """
    Stable Perlin-like noise generator using gradients and smooth interpolation.
    Note: keep grid sizes moderate (<=200) for responsiveness.
    """
    np.random.seed(seed)
    h, w = shape
    # choose grid cell size
    cell = max(3, int(scale))
    # gradient grid size
    gy = h // cell + 2
    gx = w // cell + 2
    # random gradient vectors
    grad_x = np.random.randn(gy, gx)
    grad_y = np.random.randn(gy, gx)
    norm = np.sqrt(grad_x**2 + grad_y**2) + 1e-9
    grad_x /= norm
    grad_y /= norm

    # coordinate arrays
    ys = np.linspace(0, (gy - 2), h, endpoint=False)
    xs = np.linspace(0, (gx - 2), w, endpoint=False)
    y0 = ys.astype(int)
    x0 = xs.astype(int)
    yf = ys - y0
    xf = xs - x0

    def fade(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    u = fade(xf)
    v = fade(yf)

    noise = np.zeros((h, w), dtype=float)

    # compute per-pixel using vectorized outer operations for rows/cols
    for i in range(h):
        iy = y0[i]
        vy = v[i]
        dy = yf[i]
        g00x = grad_x[iy, x0]
        g00y = grad_y[iy, x0]
        g10x = grad_x[iy, x0+1]
        g10y = grad_y[iy, x0+1]
        g01x = grad_x[iy+1, x0]
        g01y = grad_y[iy+1, x0]
        g11x = grad_x[iy+1, x0+1]
        g11y = grad_y[iy+1, x0+1]

        dx = xf  # vector across columns
        # dot products across row
        n00 = g00x * dx + g00y * dy
        n10 = g10x * (dx - 1) + g10y * dy
        n01 = g01x * dx + g01y * (dy - 1)
        n11 = g11x * (dx - 1) + g11y * (dy - 1)

        nx0 = n00 * (1 - u) + n10 * u
        nx1 = n01 * (1 - u) + n11 * u
        noise[i, :] = nx0 * (1 - vy) + nx1 * vy

    # normalize safely
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)
    return noise

def synth_noise(ntype, duration, fs):
    t = np.linspace(0, duration, int(duration*fs), endpoint=False)
    if ntype == "Rain":
        return np.random.randn(len(t)) * 0.2
    if ntype == "Shipping":
        freqs = [20, 50, 80]
        signal = sum([0.4*np.sin(2*np.pi*f*t) for f in freqs])
        signal += np.random.randn(len(t))*0.05
        return signal
    if ntype == "Dolphin Clicks":
        signal = np.zeros(len(t))
        num_clicks = np.random.randint(20,60)
        click_pos = np.random.choice(len(t), num_clicks, replace=False)
        for p in click_pos:
            width = max(1, int(0.001*fs))
            signal[p:p+width] += np.hanning(width) * np.random.uniform(0.5,1.0)
        return signal
    if ntype == "Seismic":
        return np.sin(2*np.pi*5*t) * np.exp(-t*1.5) + np.random.randn(len(t))*0.02
    if ntype == "Turbulence":
        return np.cumsum(np.random.randn(len(t))) * 0.0005
    return np.zeros_like(t)

# -----------------------------
# Section: Real-time Ocean Buoy Telemetry
# -----------------------------
st.header("🛰️ Real-Time Buoy Telemetry Simulator")
col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.subheader("Live Sensor Stream")
    base_temp = st.number_input("Base Temperature (°C)", value=15.0, step=0.1)
    base_sal = st.number_input("Base Salinity (PSU)", value=35.0, step=0.1)
    base_ph = st.number_input("Base pH", value=8.0, step=0.01)

with col2:
    st.subheader("Environmental Parameters")
    base_chl = st.slider("Chlorophyll proxy", 0.0, 5.0, 1.0)
    wave_amp = st.slider("Wave amplitude (m)", 0.0, 5.0, 1.0)

with col3:
    st.subheader("Pollution Controls")
    enable_spike = st.checkbox("Enable Salinity/pH spike events", value=True)
    pollution_rate = st.slider("Pollution rate", 0.0, 1.0, 0.1)

telemetry_placeholder = st.empty()
telemetry_chart = st.empty()

buffer_len = 200
time_buf = list(np.linspace(-buffer_len+1, 0, buffer_len))
temp_buf = [base_temp + np.random.randn()*0.05 for _ in range(buffer_len)]
sal_buf = [base_sal + np.random.randn()*0.02 for _ in range(buffer_len)]
ph_buf = [base_ph + np.random.randn()*0.005 for _ in range(buffer_len)]
chl_buf = [base_chl + np.random.randn()*0.01 for _ in range(buffer_len)]
wave_buf = [wave_amp * np.sin(i*0.2) + np.random.randn()*0.05 for i in range(buffer_len)]

def update_telemetry():
    global temp_buf, sal_buf, ph_buf, chl_buf, wave_buf, time_buf
    t_now = time.time()
    temp = temp_buf[-1] + np.random.randn()*0.02 + 0.001*(base_temp - temp_buf[-1])
    sal = sal_buf[-1] + np.random.randn()*0.01 + 0.0005*(base_sal - sal_buf[-1])
    ph = ph_buf[-1] + np.random.randn()*0.003 + 0.0002*(base_ph - ph_buf[-1])
    chl = max(0, chl_buf[-1] + np.random.randn()*0.01 + 0.0003*(base_chl - chl_buf[-1]))
    wave = wave_amp * np.sin(t_now*0.5) + np.random.randn()*0.05
    if enable_spike and (np.random.rand() < 0.005*max(1,pollution_rate*10)):
        sal += np.random.uniform(0.5, 3.0)
        ph -= np.random.uniform(0.1, 0.5)
    time_buf.append(time_buf[-1] + 1)
    temp_buf.append(temp); sal_buf.append(sal); ph_buf.append(ph); chl_buf.append(chl); wave_buf.append(wave)
    for buf in [time_buf, temp_buf, sal_buf, ph_buf, chl_buf, wave_buf]:
        while len(buf) > buffer_len:
            buf.pop(0)
    return temp, sal, ph, chl, wave

_ = update_telemetry()
with telemetry_placeholder.container():
    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
    tcol1.metric("Temp (°C)", f"{temp_buf[-1]:.2f}")
    tcol2.metric("Salinity (PSU)", f"{sal_buf[-1]:.2f}")
    tcol3.metric("pH", f"{ph_buf[-1]:.2f}")
    tcol4.metric("Chlorophyll proxy", f"{chl_buf[-1]:.2f}")

with telemetry_chart.container():
    fig, ax = plt.subplots(2,2, figsize=(9,6))
    ax[0,0].plot(temp_buf, label="Temp"); ax[0,0].set_title("Temperature (°C)")
    ax[0,1].plot(sal_buf, label="Salinity"); ax[0,1].set_title("Salinity (PSU)")
    ax[1,0].plot(ph_buf, label="pH"); ax[1,0].set_title("pH")
    ax[1,1].plot(chl_buf, label="Chlorophyll"); ax[1,1].set_title("Chlorophyll proxy")
    plt.tight_layout()
    st.pyplot(fig)

if time_step is not None:
    for _ in range(3 if sim_speed=="Realtime" else 10):
        _ = update_telemetry()
        # quick metric update
        telemetry_placeholder.metric("Temp (°C)", f"{temp_buf[-1]:.2f}")
        telemetry_placeholder.metric("Salinity (PSU)", f"{sal_buf[-1]:.2f}")
        telemetry_placeholder.metric("pH", f"{ph_buf[-1]:.2f}")
        telemetry_placeholder.metric("Chlorophyll proxy", f"{chl_buf[-1]:.2f}")
        time.sleep(time_step/5 if sim_speed!="Paused" else 0)

# -----------------------------
# Section: 3D Bathymetry and Remote Sensing Layers
# -----------------------------
st.header("🗺️ 3D Bathymetry & Remote-Sensing Layers")
col_a, col_b = st.columns([1,1])

with col_a:
    size = st.slider("Bathymetry grid size (pixels)", 50, 150, 100)
    roughness = st.slider("Seafloor roughness (scale)", 1, 8, 3)
    seed_bathy = st.number_input("Bathymetry seed", value=1, step=1)
    bathy = perlin_noise((size,size), scale=roughness, seed=seed_bathy)
    depth_scale = st.slider("Max depth (m)", 100, 11000, 4000)
    depth_map = (bathy * depth_scale) * -1  # negative for depth
    fig3d = go.Figure(data=[go.Surface(z=depth_map, colorscale='Viridis', showscale=True)])
    fig3d.update_layout(title="Procedural Bathymetry (3D Surface)", autosize=True, height=450)
    st.plotly_chart(fig3d, use_container_width=True)

with col_b:
    st.subheader("Procedural Remote Sensing Layers")
    rs_seed = st.number_input("Remote-sensing seed", value=10, step=1)
    rs_size = size
    sst = perlin_noise((rs_size, rs_size), scale=3.0, seed=rs_seed) * 10 + 15  # SST in °C
    chl = perlin_noise((rs_size, rs_size), scale=2.0, seed=rs_seed+3) * 3  # chlorophyll proxy
    ssh = perlin_noise((rs_size, rs_size), scale=4.0, seed=rs_seed+7) * 0.5  # sea surface height m
    col_choice = st.selectbox("Layer to display", ["SST", "Chlorophyll", "Sea Surface Height"])
    layer = {"SST": sst, "Chlorophyll": chl, "Sea Surface Height": ssh}[col_choice]
    fig_rs = px.imshow(layer, color_continuous_scale="thermal" if col_choice=="SST" else "viridis")
    fig_rs.update_layout(height=450, title=f"Procedural {col_choice}")
    st.plotly_chart(fig_rs, use_container_width=True)

# -----------------------------
# Section: ROV/AUV Navigation Simulator
# -----------------------------
st.header("🤖 ROV / AUV Navigation Simulator")
rov_col1, rov_col2 = st.columns([1,1])
with rov_col1:
    st.subheader("Control Panel")
    target_depth = st.slider("Target Depth (m)", 0, int(abs(depth_map.min())), 100)
    heading = st.slider("Heading (°)", 0, 359, 0)
    pitch = st.slider("Pitch (°)", -30, 30, 0)
    thrust = st.slider("Thrust (%)", 0, 100, 50)
    battery_setting = st.slider("Initial Battery (%)", 0, 100, 80)
    simulate_rov = st.button("Run ROV step")
with rov_col2:
    st.subheader("ROV Telemetry")
    rov_depth = st.empty()
    rov_heading = st.empty()
    rov_battery = st.empty()
    rov_sonar = st.empty()

if "rov_state" not in st.session_state:
    st.session_state.rov_state = {"depth": 0.0, "heading": 0.0, "battery": battery_setting, "x": size//2, "y": size//2}

def rov_step(target_depth, heading, pitch, thrust):
    state = st.session_state.rov_state
    depth_change = thrust/100.0 * math.sin(math.radians(pitch)) * 5.0
    state["heading"] += (heading - state["heading"]) * 0.1
    state["depth"] += depth_change
    state["depth"] = max(0, state["depth"])
    state["battery"] -= abs(thrust)/100.0 * 0.05 + 0.01
    state["battery"] = max(0, state["battery"])
    dx = math.cos(math.radians(state["heading"])) * thrust/100.0 * 0.5
    dy = math.sin(math.radians(state["heading"])) * thrust/100.0 * 0.5
    state["x"] = np.clip(state["x"] + dx, 0, size-1)
    state["y"] = np.clip(state["y"] + dy, 0, size-1)
    seabed_depth = -depth_map[int(max(0,min(size-1,state["y"]))), int(max(0,min(size-1,state["x"])))]
    distance_to_seafloor = max(0, seabed_depth - state["depth"])
    st.session_state.rov_state = state
    return state, distance_to_seafloor

if simulate_rov:
    state, sonar_dist = rov_step(target_depth, heading, pitch, thrust)
    rov_depth.write(f"Depth: {state['depth']:.2f} m")
    rov_heading.write(f"Heading: {state['heading']:.1f} °")
    rov_battery.write(f"Battery: {state['battery']:.1f} %")
    rov_sonar.write(f"Sonar distance to seafloor: {sonar_dist:.2f} m")

fig_pos = px.imshow(depth_map, origin='lower', color_continuous_scale='Blues', title="ROV Position (2D Bathymetry)")
if "rov_state" in st.session_state:
    s = st.session_state.rov_state
    fig_pos.add_scatter(x=[s["x"]], y=[s["y"]], mode="markers", marker=dict(size=8, color="red"), name="ROV")
st.plotly_chart(fig_pos, use_container_width=True)

# -----------------------------
# Virtual CTD profiler
# -----------------------------
st.header("📊 Virtual CTD Profiling")
ctd_col1, ctd_col2 = st.columns([1,1])
with ctd_col1:
    max_profile_depth = st.slider("CTD max depth for profile (m)", 50, int(abs(depth_map.min())), 500)
    profile_steps = st.slider("Profile steps", 10, 500, 100)
    z = np.linspace(0, max_profile_depth, profile_steps)
    thermocline_depth = st.slider("Thermocline depth (m)", 10, 200, 50)
    temp_surface = base_temp + np.random.randn()*0.2
    temp_bottom = base_temp - np.random.uniform(2.0,6.0)
    temp_profile = temp_surface + (temp_bottom - temp_surface) * (z / max_profile_depth) + 2.5 * np.exp(-((z-thermocline_depth)/10.0)**2)
    sal_profile = base_sal + (np.random.randn(profile_steps)*0.05) + 0.02*(z/max_profile_depth)
    sound_profile = sound_speed_unesco(temp_profile, sal_profile, z)
    density_profile = density_approx(temp_profile, sal_profile)
    fig_ctd, ax_ctd = plt.subplots(1,3, figsize=(12,4))
    ax_ctd[0].plot(temp_profile, z); ax_ctd[0].invert_yaxis(); ax_ctd[0].set_xlabel("Temp °C"); ax_ctd[0].set_title("T vs Depth")
    ax_ctd[1].plot(sal_profile, z); ax_ctd[1].invert_yaxis(); ax_ctd[1].set_xlabel("Sal PSU"); ax_ctd[1].set_title("S vs Depth")
    ax_ctd[2].plot(sound_profile, z); ax_ctd[2].invert_yaxis(); ax_ctd[2].set_xlabel("Sound speed m/s"); ax_ctd[2].set_title("Sound speed vs Depth")
    st.pyplot(fig_ctd)
with ctd_col2:
    st.subheader("Calculated Profiles (summary)")
    st.write(f"Mean Temp: {np.mean(temp_profile):.2f} °C")
    st.write(f"Mean Salinity: {np.mean(sal_profile):.2f} PSU")
    st.write(f"Min Sound Speed: {np.min(sound_profile):.2f} m/s  |  Max: {np.max(sound_profile):.2f} m/s")
    if st.button("Export CTD profile (CSV)"):
        df = pd.DataFrame({"depth_m":z, "temp_C":temp_profile, "sal_PSU":sal_profile, "sound_m_s":sound_profile, "density_kg_m3":density_profile})
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CTD CSV", csv, "ctd_profile.csv", "text/csv")

# -----------------------------
# Pollution Pathway Visualizer (Advection-Diffusion)
# -----------------------------
st.header("🛢️ Pollution Pathway Visualizer (Advection–Diffusion)")
pp_col1, pp_col2 = st.columns([1,1])
with pp_col1:
    sim_grid = st.slider("Pollution sim grid size", 50, 150, 80)
    adv_strength = st.slider("Advection strength", 0.0, 2.0, 0.5)
    diff_coeff = st.slider("Diffusion coefficient", 0.0, 1.0, 0.2)
    current_angle = st.slider("Current direction (°)", 0, 359, 45)
    source_x = st.slider("Source X position", 0, sim_grid-1, sim_grid//2)
    source_y = st.slider("Source Y position", 0, sim_grid-1, sim_grid//2)
    steps_sim = st.slider("Simulation steps", 1, 200, 80)
with pp_col2:
    st.subheader("Run Simulation")
    grid = np.zeros((sim_grid, sim_grid))
    grid[source_y, source_x] = 1.0
    angle_rad = math.radians(current_angle)
    vx = adv_strength * math.cos(angle_rad)
    vy = adv_strength * math.sin(angle_rad)
    for t_step in range(steps_sim):
        lap = (np.roll(grid,1,0) + np.roll(grid,-1,0) + np.roll(grid,1,1) + np.roll(grid,-1,1) - 4*grid)
        grid = grid + diff_coeff * lap + 0.01 * np.random.randn(*grid.shape)
        shiftx = int(np.sign(vx))
        shifty = int(np.sign(vy))
        if shiftx != 0:
            grid = np.roll(grid, shiftx, axis=1)
        if shifty != 0:
            grid = np.roll(grid, shifty, axis=0)
        grid *= 0.999
    fig_pp = px.imshow(grid, color_continuous_scale="inferno", title="Pollution Concentration")
    st.plotly_chart(fig_pp, use_container_width=True)

# -----------------------------
# Underwater Noise Comparison Dashboard
# -----------------------------
st.header("🔊 Underwater Noise Comparison Dashboard")
noise_types = ["Rain", "Shipping", "Dolphin Clicks", "Seismic", "Turbulence"]
selected_noises = st.multiselect("Select noise types to compare", noise_types, default=["Rain", "Shipping", "Dolphin Clicks"])
duration = st.slider("Duration (s) for spectrogram", 1, 5, 2)
fs = 4000
spec_cols = st.columns(len(selected_noises) if selected_noises else 1)
for i, ntype in enumerate(selected_noises):
    sig = synth_noise(ntype, duration, fs)
    f, ts, Sxx = spectrogram(sig, fs, nperseg=256, noverlap=128)
    fig_spec = plt.figure(figsize=(4,3))
    plt.pcolormesh(ts, f, 10*np.log10(Sxx+1e-9))
    plt.ylabel('Freq [Hz]'); plt.xlabel('Time [sec]')
    plt.title(f"Spectrogram: {ntype}")
    spec_cols[i].pyplot(fig_spec)

# -----------------------------
# Light Attenuation & Photosynthesis
# -----------------------------
st.header("☀️ Light Attenuation & Primary Productivity")
la_col1, la_col2 = st.columns([1,1])
with la_col1:
    I0 = st.number_input("Surface irradiance I0", value=2000.0)
    k = st.slider("Attenuation coeff k (m⁻¹)", 0.01, 1.0, 0.1)
    z_depths = np.linspace(0, 200, 201)
    I_z = I0 * np.exp(-k * z_depths)
    fig_la = plt.figure(figsize=(6,3))
    plt.plot(I_z, z_depths); plt.gca().invert_yaxis()
    plt.xlabel("I (units)"); plt.ylabel("Depth (m)"); plt.title("Beer-Lambert Light Attenuation")
    st.pyplot(fig_la)
with la_col2:
    Pmax = st.number_input("Max primary productivity Pmax (mgC m⁻² d⁻¹)", value=200.0)
    Ik = st.number_input("Saturation irradiance Ik", value=100.0)
    nutrient_lim = st.slider("Nutrient limitation factor", 0.0, 1.0, 0.8)
    P_z = Pmax * (I_z/(I_z+Ik)) * nutrient_lim
    fig_pprod = plt.figure(figsize=(6,3))
    plt.plot(P_z, z_depths); plt.gca().invert_yaxis(); plt.xlabel("P (mgC m⁻² d⁻¹)"); plt.title("Primary Productivity vs Depth")
    st.pyplot(fig_pprod)

# -----------------------------
# Marine Ecosystem Simulator
# -----------------------------
st.header("🌱 Marine Ecosystem Simulator (Phytoplankton–Zooplankton)")
eco_col1, eco_col2 = st.columns([1,1])
with eco_col1:
    r = st.number_input("Phyto growth rate r", value=0.6, step=0.05)
    K = st.number_input("Carrying capacity K", value=1000.0, step=10.0)
    grazing = st.number_input("Grazing rate", value=0.02, step=0.01)
    z_mort = st.number_input("Zooplankton mortality", value=0.1, step=0.01)
    sim_days = st.slider("Simulation days", 10, 365, 120)
with eco_col2:
    P0 = st.number_input("Initial phytoplankton", value=100.0)
    Z0 = st.number_input("Initial zooplankton", value=20.0)
    P = np.zeros(sim_days); Z = np.zeros(sim_days); t = np.arange(sim_days)
    P[0], Z[0] = P0, Z0
    for day in range(1, sim_days):
        dP = r * P[day-1] * (1 - P[day-1]/K) - grazing * P[day-1] * Z[day-1]
        dZ = grazing * P[day-1] * Z[day-1] * 0.1 - z_mort * Z[day-1]
        P[day] = max(0, P[day-1] + dP)
        Z[day] = max(0, Z[day-1] + dZ)
    fig_eco = plt.figure(figsize=(8,3))
    plt.plot(t, P, label="Phytoplankton")
    plt.plot(t, Z, label="Zooplankton")
    plt.legend(); plt.xlabel("Days"); plt.title("Ecosystem Simulation")
    st.pyplot(fig_eco)

# -----------------------------
# Tectonic Plates Animation (simplified)
# -----------------------------
st.header("🌋 Tectonic Plates & Oceanic Crust Simulator")
tp_col1, tp_col2 = st.columns([1,1])
with tp_col1:
    plates = st.slider("Number of tectonic plates (sim)", 2, 8, 4)
    plate_seed = st.number_input("Plate seed", value=7, step=1)
    plate_move = st.slider("Plate movement scale", 0.0, 5.0, 1.0)
    base_plate_map = perlin_noise((size,size), scale=3.0, seed=plate_seed)
    rng = np.random.RandomState(plate_seed)
    centers = rng.randint(0, size, size=(plates, 2))
    plate_map = np.zeros((size,size), dtype=int)
    coords = np.indices((size,size)).transpose(1,2,0).reshape(-1,2)
    for idx, (x,y) in enumerate(coords):
        dists = np.sum((centers - np.array([y,x]))**2, axis=1)
        plate_map[y,x] = np.argmin(dists)
    fig_plate = px.imshow(plate_map, color_continuous_scale=px.colors.qualitative.Dark24, title="Tectonic Plates (synthetic)")
    st.plotly_chart(fig_plate, use_container_width=True)
with tp_col2:
    st.write("Simulated plate interactions:")
    st.write("- Divergent margins: where adjacent plates move apart (visualized by color gradients).")
    st.write("- Convergent: collisions generate 'trenches' in bathymetry (procedural).")
    if st.button("Apply plate movement step"):
        for p in range(plates):
            mask = (plate_map==p)
            shift = (np.random.randn()*0.5 + plate_move*0.1)
            depth_map[mask] += shift
        st.success("Applied plate movement to bathymetry!")

# -----------------------------
# EEZ Visualizer & UNCLOS Quiz
# -----------------------------
st.header("⚖️ EEZ Visualizer & Interactive UNCLOS Learning")
eez_col1, eez_col2 = st.columns([1,1])
with eez_col1:
    island_x = st.slider("Island center X", 0, size-1, size//3)
    island_y = st.slider("Island center Y", 0, size-1, size//3)
    island_radius = st.slider("Island radius (pixels)", 1, size//3, 8)
    eez_dist = st.slider("EEZ radius (pixels, visual)", 12, size-1, 50)
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    dist = np.sqrt((xx - island_x)**2 + (yy - island_y)**2)
    island_mask = dist <= island_radius
    eez_mask = dist <= eez_dist
    canvas = np.zeros((size,size,3))
    canvas[:,:,0] = island_mask * 1.0
    canvas[:,:,1] = eez_mask * 0.5
    fig_eez = px.imshow(canvas, title="Island (red) & EEZ (green tint)", origin='lower')
    st.plotly_chart(fig_eez, use_container_width=True)
with eez_col2:
    st.subheader("UNCLOS Mini-Quiz")
    q1 = st.radio("Territorial waters extend up to:", ["12 nautical miles", "200 nautical miles", "24 nautical miles"], index=0)
    q2 = st.radio("EEZ grants exclusive rights over:", ["Fishing and resources", "Military bases", "Airspace control"], index=0)
    if st.button("Check answers"):
        score = 0
        score += 1 if q1=="12 nautical miles" else 0
        score += 1 if q2=="Fishing and resources" else 0
        st.info(f"You scored {score}/2")

# -----------------------------
# Instrumentation Failure Simulator
# -----------------------------
st.header("🔧 Instrumentation Failure Simulator")
if "fail_state" not in st.session_state:
    st.session_state.fail_state = {"ctd_drift": False, "winch_fail": False, "sedtrap_overflow": False}
fail_col1, fail_col2 = st.columns([1,1])
with fail_col1:
    if st.button("Inject random failure"):
        choice = np.random.choice(["ctd_drift", "winch_fail", "sedtrap_overflow"])
        st.session_state.fail_state[choice] = True
        st.warning(f"Injected failure: {choice}")
    if st.button("Reset failures"):
        st.session_state.fail_state = {"ctd_drift": False, "winch_fail": False, "sedtrap_overflow": False}
        st.success("Failures reset")
with fail_col2:
    st.write("Current failure states:")
    st.json(st.session_state.fail_state)
    if st.session_state.fail_state["ctd_drift"]:
        st.write("CTD sensor drift detected — simulated temperature offset applied.")
        temp_buf[-1] += np.random.uniform(-0.5, 0.5)
    if st.session_state.fail_state["winch_fail"]:
        st.write("Winch cable tension abnormal — limit winch depth commands.")
    if st.session_state.fail_state["sedtrap_overflow"]:
        st.write("Sediment trap overflow — deposition rate spikes.")

# -----------------------------
# Underwater Acoustic Communication Demo
# -----------------------------
st.header("📡 Underwater Acoustic Communication Demo")
msg = st.text_input("Type a short message to send (ASCII characters only)", value="HELLO OCEAN")
tx_freq = st.slider("Carrier frequency (Hz)", 1000, 50000, 12000)
snr = st.slider("SNR (simulated dB)", 0, 40, 20)
if st.button("Transmit message"):
    bits = ''.join([format(ord(c),'08b') for c in msg])
    bit_dur = 0.001
    fs_comm = 40000
    t_comm = np.arange(0, len(bits)*bit_dur, 1/fs_comm)
    waveform = np.zeros_like(t_comm)
    for i, b in enumerate(bits):
        seg = (t_comm >= i*bit_dur) & (t_comm < (i+1)*bit_dur)
        if b == '1':
            waveform[seg] = np.sin(2*np.pi*tx_freq*t_comm[seg])
        else:
            waveform[seg] = 0
    noise = synth_noise("Turbulence", len(t_comm)/fs_comm, fs_comm) + 0.2*synth_noise("Shipping", len(t_comm)/fs_comm, fs_comm)
    sig_power = np.mean(waveform**2) + 1e-12
    noise_power = np.mean(noise**2) + 1e-12
    desired_noise_power = sig_power / (10**(snr/10))
    if noise_power > 0:
        noise *= np.sqrt(desired_noise_power / (noise_power + 1e-12))
    rx = waveform + noise
    envelope = np.abs(np.convolve(rx, np.ones(int(fs_comm*bit_dur))/int(fs_comm*bit_dur), mode='same'))
    thresh = envelope.mean()*1.5
    rec_bits = []
    for i in range(len(bits)):
        seg_idx = int((i+0.5)*bit_dur*fs_comm)
        seg_idx = min(seg_idx, len(envelope)-1)
        rec_bits.append('1' if envelope[seg_idx] > thresh else '0')
    rec_chars = ''.join([chr(int(''.join(rec_bits[i:i+8]),2)) for i in range(0,len(rec_bits),8)])
    st.write("Transmitted message:", msg)
    st.write("Received message (simulated):", rec_chars)
    fig_tx, ax_tx = plt.subplots(2,1,figsize=(8,4))
    ax_tx[0].plot(waveform[:2000]); ax_tx[0].set_title("Transmitted waveform (first samples)")
    f_rx, tt, Sxx_rx = spectrogram(rx, fs_comm, nperseg=512, noverlap=256)
    ax_tx[1].pcolormesh(tt, f_rx, 10*np.log10(Sxx_rx+1e-9)); ax_tx[1].set_ylabel("Freq (Hz)")
    st.pyplot(fig_tx)

# -----------------------------
# Footer and deployment tips
# -----------------------------
st.markdown("---")
st.success("OceanSense Advanced ready — explore modules and show the professor a working demo!")
