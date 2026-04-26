import streamlit as st
from streamlit_folium import st_folium
import folium
import osmnx as ox
import networkx as nx
import random
import time
import pandas as pd
import numpy as np
import os
import logging

try:
    from tensorflow import keras
except ImportError:
    keras = None

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide", page_title="Ambulance Routing Dashboard", page_icon="+")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    /* ---- Global ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1rem; }

    /* ---- Header ---- */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
        padding: 1.4rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(15, 52, 96, 0.35);
        border: 1px solid rgba(255,255,255,0.08);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(231,76,60,0.08) 0%, transparent 70%);
        animation: headerGlow 6s ease-in-out infinite;
    }
    @keyframes headerGlow {
        0%, 100% { transform: translate(0, 0); }
        50% { transform: translate(5%, 5%); }
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }
    .main-header p {
        margin: 0.4rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.75;
        font-weight: 400;
        letter-spacing: 0.5px;
        position: relative;
        z-index: 1;
    }
    .header-badge {
        display: inline-block;
        background: rgba(231,76,60,0.2);
        border: 1px solid rgba(231,76,60,0.4);
        color: #e74c3c;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }

    /* ---- Metric Cards ---- */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fc 0%, #eef1f8 100%);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748b;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e293b;
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f1f5f9;
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* ---- Buttons ---- */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
        border: 1px solid #e2e8f0;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        border: none;
        color: white;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #c0392b, #a93226);
    }

    /* ---- DataFrames ---- */
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1);
    }

    /* ---- Progress Bar ---- */
    .stProgress > div > div {
        background: linear-gradient(90deg, #e74c3c, #f39c12);
        border-radius: 10px;
    }
    .stProgress > div {
        background: #e2e8f0;
        border-radius: 10px;
    }

    /* ---- Expander ---- */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.9rem;
        border-radius: 10px;
    }

    /* ---- Slider ---- */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #e74c3c;
        border: 2px solid white;
        box-shadow: 0 2px 8px rgba(231,76,60,0.3);
    }
    .stSlider [data-baseweb="slider"] [data-testid="stTickBar"] {
        background: linear-gradient(90deg, #e74c3c, #f39c12);
    }

    /* ---- Fleet Card (sidebar) ---- */
    .fleet-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .fleet-card .fc-id {
        font-weight: 700;
        font-size: 0.9rem;
    }
    .fleet-card .fc-status {
        font-size: 0.75rem;
        padding: 0.15rem 0.5rem;
        border-radius: 20px;
        font-weight: 600;
    }
    .fc-idle { background: rgba(148,163,184,0.2); color: #94a3b8; }
    .fc-enroute { background: rgba(46,204,113,0.2); color: #2ecc71; }
    .fc-arrived { background: rgba(52,152,219,0.2); color: #3498db; }

    /* ---- Section Headers ---- */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1.2rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .section-header h3 {
        margin: 0;
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
    }

    /* ---- Alert Boxes ---- */
    .stAlert { border-radius: 10px; }

    /* ---- Legend ---- */
    .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.6rem;
        background: #f8f9fc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem;
    }
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-header">'
    '<span class="header-badge">● LIVE</span>'
    "<h1>Smart Ambulance Routing</h1>"
    "<p>Real-Time Emergency Navigation — Anna Nagar, Chennai</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ================================================================
# HELPER: safe edge attribute access (works for Graph & MultiGraph)
# ================================================================

def _edge_attr(G, u, v, attr, default=0.0):
    """Safely get an edge attribute regardless of Graph vs MultiGraph."""
    data = G.get_edge_data(u, v)
    if data is None:
        return default
    # Regular Graph: data is {attr: val, ...}
    if attr in data:
        try:
            return float(data[attr])
        except (TypeError, ValueError):
            return default
    # MultiGraph: data is {0: {attr: val}, 1: {...}, ...}
    vals = []
    for v_data in data.values():
        if isinstance(v_data, dict) and attr in v_data:
            try:
                vals.append(float(v_data[attr]))
            except (TypeError, ValueError):
                continue
    return min(vals) if vals else default


def _edge_attrs(G, u, v):
    """Return a flat dict of edge attributes (first key for MultiGraph)."""
    data = G.get_edge_data(u, v)
    if data is None:
        return {}
    if "length" in data:
        return dict(data)
    for v_data in data.values():
        if isinstance(v_data, dict):
            return dict(v_data)
    return {}


# ================================================================
# LSTM MODEL LOADING & TRAFFIC DATA
# ================================================================
TRAFFIC_CSV = "_traffic_2026.csv"
MODEL_PATH = "lstm_junction1_model.h5"
SEQUENCE_LENGTH = 24  # 24 hourly steps


@st.cache_resource
def load_lstm_model():
    if keras is None:
        logger.warning("TensorFlow not installed — LSTM predictions disabled.")
        return None
    try:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH, compile=False)
            logger.info("LSTM model loaded successfully.")
            return model
        else:
            logger.warning("LSTM model file not found at %s", MODEL_PATH)
            return None
    except Exception as e:
        logger.error("Failed to load LSTM model: %s", e)
        return None


@st.cache_data
def load_traffic_data():
    """Load and preprocess the traffic CSV for LSTM inference."""
    if not os.path.exists(TRAFFIC_CSV):
        logger.warning("Traffic CSV not found at %s", TRAFFIC_CSV)
        return None, None
    try:
        df = pd.read_csv(TRAFFIC_CSV, parse_dates=["DateTime"])
        # Normalize vehicle counts per junction to [0, 1]
        max_vehicles = df["Vehicles"].max()
        if max_vehicles == 0:
            max_vehicles = 1
        df["Vehicles_norm"] = df["Vehicles"] / max_vehicles
        return df, max_vehicles
    except Exception as e:
        logger.error("Failed to load traffic CSV: %s", e)
        return None, None


def predict_congestion(model, traffic_df, max_vehicles, num_predictions):
    """Use the LSTM model with real traffic data to predict congestion levels."""
    if model is None or traffic_df is None:
        return np.random.choice([0.3, 0.5, 0.8], size=num_predictions)

    try:
        # Build sequences from the most recent traffic data for junction 1
        j1 = traffic_df[traffic_df["Junction"] == 1]["Vehicles_norm"].values
        if len(j1) < SEQUENCE_LENGTH:
            # Pad with zeros if not enough data
            j1 = np.pad(j1, (SEQUENCE_LENGTH - len(j1), 0), constant_values=0.0)

        # Use the last SEQUENCE_LENGTH values as the base sequence
        base_seq = j1[-SEQUENCE_LENGTH:].astype("float32")

        # Create a batch with slight random perturbations to simulate different edges
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.05, size=(num_predictions, SEQUENCE_LENGTH))
        batch = np.clip(base_seq[np.newaxis, :] + noise, 0, 1).astype("float32")
        batch = batch.reshape(num_predictions, SEQUENCE_LENGTH, 1)

        predictions = model.predict(batch, verbose=0, batch_size=min(128, num_predictions)).flatten()
        # Clamp to [0, 1]
        predictions = np.clip(predictions, 0.0, 1.0)
        return predictions
    except Exception as e:
        logger.error("LSTM prediction failed: %s", e)
        return np.random.choice([0.3, 0.5, 0.8], size=num_predictions)


def get_congestion_factor(prediction):
    """Map a [0,1] prediction to a congestion multiplier."""
    if prediction < 0.4:
        return 1.0
    elif prediction < 0.7:
        return 1.5
    else:
        return 2.0


lstm_model = load_lstm_model()
traffic_df, max_vehicles = load_traffic_data()


# ================================================================
# GRAPH LOADING
# ================================================================
@st.cache_resource
def load_graph():
    """Download the Anna Nagar road network and return an undirected graph."""
    try:
        G = ox.graph_from_place(
            "Anna Nagar, Chennai, India", network_type="drive"
        )
        G = G.to_undirected()
        logger.info(
            "Graph loaded: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges()
        )
        return G
    except Exception as e:
        st.error(f"Failed to load map data: {e}")
        logger.error("Graph loading failed: %s", e)
        return None


# ================================================================
# ALTERNATE ROUTE BUILDER
# ================================================================
def build_alternate_routes(G, primary_route, start_node, destination, max_alternates=3):
    """Build alternate routes by temporarily removing edges from the primary route."""
    alternates = []
    limit = min(len(primary_route) - 1, 6)
    for i in range(1, limit):
        u, v = primary_route[i], primary_route[i + 1]
        if not G.has_edge(u, v):
            continue
        saved = _edge_attrs(G, u, v)
        G.remove_edge(u, v)
        try:
            alt = nx.shortest_path(G, start_node, destination, weight="dynamic_weight")
            if alt not in alternates and alt != primary_route:
                alternates.append(alt)
        except nx.NetworkXNoPath:
            pass
        G.add_edge(u, v, **saved)
        if len(alternates) >= max_alternates:
            break
    return alternates


# ================================================================
# ROUTE COMPUTATION
# ================================================================
def calculate_routes(seed, _lstm_model=None):
    """Load graph, apply LSTM congestion weights, fetch hospitals."""
    G = load_graph()
    if G is None:
        return None

    num_edges = G.number_of_edges()

    # --- Apply congestion weights using LSTM + real traffic data ---
    predictions = predict_congestion(_lstm_model, traffic_df, max_vehicles, num_edges)

    for idx, (u, v, data) in enumerate(G.edges(data=True)):
        length = data.get("length", 100.0)
        try:
            length = float(length)
        except (TypeError, ValueError):
            length = 100.0
        pred = float(predictions[idx]) if idx < len(predictions) else 0.5
        cf = get_congestion_factor(pred)
        data["dynamic_weight"] = length * cf
        data["congestion_factor"] = cf

    # --- Fetch hospitals ---
    hospital_nodes = []
    hospital_coords = []
    try:
        hospital_gdf = ox.features_from_place(
            "Anna Nagar, Chennai, India", tags={"amenity": "hospital"}
        )
        for geom in hospital_gdf.geometry:
            try:
                lon, lat = geom.centroid.x, geom.centroid.y
                nearest = ox.distance.nearest_nodes(G, lon, lat)
                if nearest not in hospital_nodes:
                    hospital_nodes.append(nearest)
                    hospital_coords.append((lat, lon))
            except (AttributeError, ValueError):
                continue
    except Exception as e:
        logger.warning("Could not fetch hospitals from OSM: %s", e)

    # Fallback: random nodes as hospitals
    if not hospital_nodes:
        nodes = list(G.nodes)
        hospital_nodes = random.sample(nodes, min(5, len(nodes)))
        hospital_coords = [
            (G.nodes[n]["y"], G.nodes[n]["x"]) for n in hospital_nodes
        ]

    return {
        "G": G,
        "hospital_nodes": hospital_nodes,
        "hospital_coords": hospital_coords,
    }


def compute_route_for_ambulance(scenario_data, ambulance_node):
    """Select nearest hospital and compute optimal + alternate routes."""
    if scenario_data is None:
        return None

    G = scenario_data["G"]
    hospital_nodes = scenario_data["hospital_nodes"]

    distances = {}
    for h in hospital_nodes:
        try:
            distances[h] = nx.shortest_path_length(
                G, ambulance_node, h, weight="dynamic_weight"
            )
        except nx.NetworkXNoPath:
            continue

    if not distances:
        return None

    destination = min(distances, key=distances.get)

    try:
        optimal_route = nx.shortest_path(
            G, ambulance_node, destination, weight="dynamic_weight"
        )
    except nx.NetworkXNoPath:
        return None

    lats = [G.nodes[n]["y"] for n in optimal_route]
    lons = [G.nodes[n]["x"] for n in optimal_route]
    center = [(min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2]

    alternate_routes = build_alternate_routes(
        G, optimal_route, ambulance_node, destination
    )

    return {
        "G": G,
        "start": ambulance_node,
        "destination": destination,
        "hospital_nodes": scenario_data["hospital_nodes"],
        "hospital_coords": scenario_data["hospital_coords"],
        "distances": distances,
        "optimal_route": optimal_route,
        "alternate_routes": alternate_routes,
        "center": center,
    }


# ================================================================
# STATE MANAGEMENT HELPERS
# ================================================================
def sync_ambulance_state(amb):
    """Single source of truth for node / step / status transitions."""
    if amb["routes_data"] is None:
        amb["status"] = "Idle"
        amb["auto_drive"] = False
        return

    route = amb["routes_data"].get("optimal_route", [])
    if not route:
        amb["status"] = "Idle"
        amb["auto_drive"] = False
        amb["routes_data"] = None
        return

    last_idx = len(route) - 1
    amb["step"] = max(0, min(int(amb["step"]), last_idx))
    amb["node"] = route[amb["step"]]
    amb["routes_data"]["start"] = amb["node"]

    if amb["step"] >= last_idx:
        amb["step"] = last_idx
        amb["node"] = route[-1]
        amb["routes_data"]["start"] = amb["node"]
        amb["status"] = "Arrived"
        amb["auto_drive"] = False
    else:
        amb["status"] = "En Route"


def assign_emergency_to_ambulance(ambulance_id, scenario_data, pickup_node):
    """Assign emergency: Phase 1 = ambulance → pickup."""
    amb = st.session_state.fleet[ambulance_id]
    ambulance_node = amb["node"]
    G = scenario_data["G"]

    try:
        route_to_pickup = nx.shortest_path(
            G, ambulance_node, pickup_node, weight="dynamic_weight"
        )
    except nx.NetworkXNoPath:
        return False

    amb["routes_data"] = {
        "G": G,
        "start": ambulance_node,
        "destination": pickup_node,
        "hospital_nodes": scenario_data["hospital_nodes"],
        "hospital_coords": scenario_data["hospital_coords"],
        "optimal_route": route_to_pickup,
        "alternate_routes": [],
        "center": [
            (G.nodes[ambulance_node]["y"] + G.nodes[pickup_node]["y"]) / 2,
            (G.nodes[ambulance_node]["x"] + G.nodes[pickup_node]["x"]) / 2,
        ],
    }
    amb["step"] = 0
    amb["auto_drive"] = False
    amb["status"] = "En Route"
    amb["phase"] = "ToPickup"
    amb["pickup_node"] = pickup_node
    amb["destination_hospital"] = None
    amb["reroute_count"] = 0
    amb["event_log"] = []
    amb["slider_override"] = True
    return True


# ================================================================
# FLEET INITIALIZATION
# ================================================================
if "fleet" not in st.session_state:
    G_init = load_graph()
    if G_init:
        nodes = list(G_init.nodes)
        st.session_state.fleet = {}
        for i in range(1, 6):
            st.session_state.fleet[f"A{i}"] = {
                "routes_data": None,
                "node": random.choice(nodes),
                "step": 0,
                "auto_drive": False,
                "status": "Idle",
                "reroute_count": 0,
                "event_log": [],
                "slider_override": False,
                "phase": None,
                "pickup_node": None,
                "destination_hospital": None,
            }
    else:
        st.session_state.fleet = {}

if "last_move_ts" not in st.session_state:
    st.session_state.last_move_ts = 0.0

if "current_emergency" not in st.session_state:
    st.session_state.current_emergency = None


# ================================================================
# SIDEBAR — MODE SELECTION & EMERGENCY DISPATCH
# ================================================================
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:0.5rem 0 1rem 0;">'
        '<div style="width:40px;height:40px;background:linear-gradient(135deg,#e74c3c,#c0392b);'
        'border-radius:10px;display:inline-flex;align-items:center;justify-content:center;'
        'color:white;font-size:1.2rem;font-weight:800;">+</div><br>'
        '<span style="font-size:1.1rem;font-weight:800;letter-spacing:-0.5px;">Command Center</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    mode = st.radio("View Mode", ["Admin Dashboard", "Ambulance Panel"])

    if mode == "Ambulance Panel":
        # Build descriptive labels: "A1 — Idle", "A2 — En Route", etc.
        amb_options = {
            f"{aid} — {a['status']}": aid
            for aid, a in st.session_state.fleet.items()
        }
        selected_label = st.selectbox("Select Unit", list(amb_options.keys()))
        selected_ambulance = amb_options[selected_label]

    st.divider()

    if st.button(
        "New Emergency Dispatch",
        type="primary",
        use_container_width=True,
    ):
        scenario_seed = int(time.time())
        scenario_data = calculate_routes(scenario_seed, lstm_model)
        if scenario_data:
            G = scenario_data["G"]
            emergency_node = random.choice(list(G.nodes))
            st.session_state.current_emergency = emergency_node

            if mode == "Ambulance Panel":
                if assign_emergency_to_ambulance(
                    selected_ambulance, scenario_data, emergency_node
                ):
                    st.success(f"{selected_ambulance} dispatched to pickup")
                else:
                    st.error(f"No route for {selected_ambulance}")
            else:
                # Admin mode: auto-dispatch nearest idle ambulance
                idle = [
                    aid
                    for aid, a in st.session_state.fleet.items()
                    if a["status"] == "Idle"
                ]
                if idle:
                    best_id, best_dist = None, float("inf")
                    for aid in idle:
                        try:
                            d = nx.shortest_path_length(
                                G,
                                st.session_state.fleet[aid]["node"],
                                emergency_node,
                                weight="dynamic_weight",
                            )
                            if d < best_dist:
                                best_dist, best_id = d, aid
                        except nx.NetworkXNoPath:
                            continue
                    if best_id and assign_emergency_to_ambulance(
                        best_id, scenario_data, emergency_node
                    ):
                        st.success(f"{best_id} dispatched (nearest idle)")
                    else:
                        st.error("No route available")
                else:
                    st.warning("All ambulances are busy")
        st.rerun()

    # Sidebar fleet summary
    st.divider()
    st.markdown(
        '<div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;'
        'letter-spacing:1px;color:#64748b;margin-bottom:0.5rem;">Fleet Status</div>',
        unsafe_allow_html=True,
    )
    for aid, a in st.session_state.fleet.items():
        status_cls = {"Idle": "fc-idle", "En Route": "fc-enroute", "Arrived": "fc-arrived"}.get(
            a["status"], "fc-idle"
        )
        icon = ""
        st.markdown(
            f'<div class="fleet-card">'
            f'<span class="fc-id">{aid}</span>'
            f'<span class="fc-status {status_cls}">{a["status"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ================================================================
# GLOBAL AUTO-MOVEMENT ENGINE
# ================================================================
for amb_id, amb in st.session_state.fleet.items():
    sync_ambulance_state(amb)
    if amb["status"] == "Arrived":
        amb["auto_drive"] = False

MOVE_INTERVAL_SEC = 0.6
now_ts = time.time()
tick_ready = (now_ts - st.session_state.last_move_ts) >= MOVE_INTERVAL_SEC

if tick_ready:
    for amb in st.session_state.fleet.values():
        if amb["routes_data"] and amb["auto_drive"]:
            route = amb["routes_data"]["optimal_route"]
            last_idx = len(route) - 1
            if amb["status"] == "Arrived" or amb["step"] >= last_idx:
                amb["auto_drive"] = False
                sync_ambulance_state(amb)
                continue
            amb["step"] += 1
            if amb["step"] >= last_idx:
                amb["auto_drive"] = False
            sync_ambulance_state(amb)
    st.session_state.last_move_ts = now_ts

# Compute AFTER tick so it reflects the just-updated state
any_auto_drive = any(
    a["auto_drive"] and a["routes_data"] and a["status"] not in ("Arrived",)
    for a in st.session_state.fleet.values()
)


# ================================================================
# ADMIN DASHBOARD MODE
# ================================================================
if mode == "Admin Dashboard":
    st.markdown(
        '<div class="section-header"><h3>Fleet Overview</h3></div>',
        unsafe_allow_html=True,
    )

    # Summary metrics row
    total = len(st.session_state.fleet)
    idle_count = sum(1 for a in st.session_state.fleet.values() if a["status"] == "Idle")
    active_count = sum(1 for a in st.session_state.fleet.values() if a["status"] == "En Route")
    arrived_count = sum(1 for a in st.session_state.fleet.values() if a["status"] == "Arrived")

    sm1, sm2, sm3, sm4 = st.columns(4)
    with sm1:
        st.metric("Total Fleet", total)
    with sm2:
        st.metric("Available", idle_count)
    with sm3:
        st.metric("Active", active_count)
    with sm4:
        st.metric("Completed", arrived_count)

    # Fleet status table
    fleet_rows = []
    for aid, a in st.session_state.fleet.items():
        if a["routes_data"] and a["status"] == "En Route":
            rlen = len(a["routes_data"]["optimal_route"])
            prog = int((a["step"] / max(rlen - 1, 1)) * 100)
        else:
            prog = 100 if a["status"] == "Arrived" else 0
        phase_label = ""
        if a["phase"] == "ToPickup":
            phase_label = "→ Pickup"
        elif a["phase"] == "ToHospital":
            phase_label = "→ Hospital"
        fleet_rows.append(
            {
                "ID": aid,
                "Status": a["status"],
                "Phase": phase_label,
                "Progress": f"{prog}%",
                "Reroutes": a["reroute_count"],
            }
        )
    st.dataframe(pd.DataFrame(fleet_rows), use_container_width=True, hide_index=True)

    # --- Admin Map ---
    G = load_graph()
    if G:
        all_lats, all_lons = [], []
        for a in st.session_state.fleet.values():
            if a["routes_data"]:
                for n in a["routes_data"]["optimal_route"]:
                    all_lats.append(G.nodes[n]["y"])
                    all_lons.append(G.nodes[n]["x"])

        center = (
            [(min(all_lats) + max(all_lats)) / 2, (min(all_lons) + max(all_lons)) / 2]
            if all_lats
            else [13.0850, 80.2101]
        )

        m = folium.Map(location=center, zoom_start=15, control_scale=True)

        # Emergency marker
        if st.session_state.current_emergency:
            en = st.session_state.current_emergency
            folium.Marker(
                location=(G.nodes[en]["y"], G.nodes[en]["x"]),
                popup="Emergency",
                icon=folium.DivIcon(
                    html=(
                        '<div style="width:18px;height:18px;background:#e74c3c;border-radius:50%;'
                        'border:3px solid #fff;box-shadow:0 2px 6px rgba(0,0,0,.4);'
                        'animation:blink 1s infinite;"></div>'
                        "<style>@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}</style>"
                    ),
                    icon_size=(18, 18),
                    icon_anchor=(9, 9),
                ),
            ).add_to(m)

        # Draw each ambulance
        for aid, a in st.session_state.fleet.items():
            if a["routes_data"] and a["status"] in ("En Route", "Arrived"):
                route = a["routes_data"]["optimal_route"]
                color = "orange" if a["phase"] == "ToPickup" else "red"
                label = "→ Pickup" if a["phase"] == "ToPickup" else "→ Hospital"

                # Old route (gray) if rerouted
                if "old_route" in a["routes_data"]:
                    pts = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in a["routes_data"]["old_route"]]
                    folium.PolyLine(pts, color="gray", weight=2, opacity=0.4).add_to(m)

                pts = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]
                folium.PolyLine(
                    pts, color=color, weight=4, opacity=0.7, tooltip=f"{aid} {label}"
                ).add_to(m)

                cur = route[a["step"]]
                icon_txt_html = (
                    '<div style="width:14px;height:14px;background:#2ecc71;border-radius:50%;'
                    'border:2px solid #fff;box-shadow:0 2px 4px rgba(0,0,0,.3);"></div>'
                    if a["step"] == len(route) - 1
                    else '<div style="width:16px;height:16px;background:#e74c3c;border-radius:50%;'
                    'border:3px solid #fff;box-shadow:0 2px 6px rgba(0,0,0,.4);"></div>'
                )
                folium.Marker(
                    location=(G.nodes[cur]["y"], G.nodes[cur]["x"]),
                    popup=f"{aid} — Step {a['step']+1}/{len(route)}",
                    icon=folium.DivIcon(
                        html=icon_txt_html,
                        icon_size=(16, 16),
                        icon_anchor=(8, 8),
                    ),
                ).add_to(m)

            elif a["status"] == "Idle" and a["node"]:
                folium.Marker(
                    location=(G.nodes[a["node"]]["y"], G.nodes[a["node"]]["x"]),
                    popup=f"{aid} — Idle",
                    icon=folium.DivIcon(
                        html=(
                            '<div style="width:12px;height:12px;background:#94a3b8;border-radius:50%;'
                            'border:2px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,.3);opacity:0.7;"></div>'
                        ),
                        icon_size=(12, 12),
                        icon_anchor=(6, 6),
                    ),
                ).add_to(m)

        st_folium(m, height=600, use_container_width=True)


# ================================================================
# AMBULANCE PANEL MODE
# ================================================================
else:
    amb = st.session_state.fleet[selected_ambulance]

    if amb["routes_data"] is None:
        st.info(
            f"{selected_ambulance} is Idle. Generate an emergency scenario to dispatch."
        )
        st.stop()

    data = amb["routes_data"]
    G = data["G"]
    route = data["optimal_route"]
    max_idx = max(len(route) - 1, 0)
    amb["step"] = max(0, min(amb["step"], max_idx))
    route_steps = max(max_idx, 1)
    progress = int((amb["step"] / route_steps) * 100)

    # ---- Analytics ----
    phase_txt = "To Pickup" if amb["phase"] == "ToPickup" else "To Hospital"
    phase_color = "#e74c3c" if amb["phase"] == "ToPickup" else "#3498db"
    st.markdown(
        f'<div class="section-header">'
        f'<h3>{selected_ambulance} — '
        f'<span style="color:{phase_color}">{phase_txt}</span></h3>'
        f'</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Progress", f"{progress}%")
    with c2:
        st.metric("Status", amb["status"])
    with c3:
        remaining = max(len(route) - amb["step"] - 1, 0)
        st.metric("Steps Left", remaining)
    with c4:
        if len(route) > 1:
            cfs = [
                _edge_attr(G, route[i], route[i + 1], "congestion_factor", 1.0)
                for i in range(len(route) - 1)
            ]
            avg_cf = float(np.mean(cfs))
        else:
            avg_cf = 1.0
        label = "Low" if avg_cf < 1.3 else ("Medium" if avg_cf < 1.8 else "High")
        st.metric("Congestion", label)

    # ---- Tabs: Hospital Analysis & Route Comparison ----
    tab1, tab2 = st.tabs(["Hospital Analysis", "Route Comparison"])

    with tab1:
        if "distances" in data and data["distances"]:
            rows = []
            for rank, (h, d) in enumerate(
                sorted(data["distances"].items(), key=lambda x: x[1]), 1
            ):
                rows.append(
                    {
                        "Rank": rank,
                        "Hospital Node": h,
                        "Distance (km)": round(d / 1000, 2),
                        "Status": "Selected"
                        if h == data.get("destination")
                        else "Available",
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Hospital analysis available after patient pickup.")

    with tab2:
        rt_rows = []
        # Optimal route distance
        opt_dist = sum(
            _edge_attr(G, route[j], route[j + 1], "dynamic_weight", 0)
            for j in range(len(route) - 1)
        )
        rt_rows.append(
            {
                "Route": "Optimal",
                "Distance (km)": round(opt_dist / 1000, 2),
                "Nodes": len(route),
                "Weights": "LSTM-based",
                "Status": "Active",
            }
        )
        for i, alt in enumerate(data.get("alternate_routes", [])):
            try:
                ad = sum(
                    _edge_attr(G, alt[j], alt[j + 1], "dynamic_weight", 0)
                    for j in range(len(alt) - 1)
                )
                rt_rows.append(
                    {
                        "Route": f"Alternate {i+1}",
                        "Distance (km)": round(ad / 1000, 2),
                        "Nodes": len(alt),
                        "Weights": "LSTM-based",
                        "Status": "Backup",
                    }
                )
            except (KeyError, IndexError):
                continue
        st.dataframe(pd.DataFrame(rt_rows), use_container_width=True, hide_index=True)


    # ---- Movement Controls ----
    st.markdown(
        '<div class="section-header"><h3>Movement Control</h3></div>',
        unsafe_allow_html=True,
    )

    # Accident simulation button
    if st.button("Simulate Accident on Route", type="secondary", use_container_width=True):
        if 0 < amb["step"] < len(route) - 1:
            current_node = route[amb["step"]]
            old_route = route.copy()

            # Calculate old metrics
            old_cf = float(np.mean([
                _edge_attr(G, old_route[i], old_route[i + 1], "congestion_factor", 1.0)
                for i in range(len(old_route) - 1)
            ])) if len(old_route) > 1 else 1.0
            old_time = sum(
                _edge_attr(G, old_route[i], old_route[i + 1], "dynamic_weight", 0)
                for i in range(amb["step"], len(old_route) - 1)
            )

            # Spike congestion on remaining route edges
            for i in range(amb["step"], len(route) - 1):
                u, v = route[i], route[i + 1]
                if G.has_edge(u, v):
                    length = _edge_attr(G, u, v, "length", 100)
                    attrs = _edge_attrs(G, u, v)
                    attrs["dynamic_weight"] = length * 3.0
                    attrs["congestion_factor"] = 3.0
                    # Update edge — works for both Graph and MultiGraph
                    nx.set_edge_attributes(G, {(u, v): attrs})

            try:
                new_route = nx.shortest_path(
                    G, current_node, data["destination"], weight="dynamic_weight"
                )
                new_cf = float(np.mean([
                    _edge_attr(G, new_route[i], new_route[i + 1], "congestion_factor", 1.0)
                    for i in range(len(new_route) - 1)
                ])) if len(new_route) > 1 else 1.0
                new_time = sum(
                    _edge_attr(G, new_route[i], new_route[i + 1], "dynamic_weight", 0)
                    for i in range(len(new_route) - 1)
                )
                time_saved = old_time - new_time

                data["old_route"] = old_route
                data["optimal_route"] = new_route
                data["alternate_routes"] = build_alternate_routes(
                    G, new_route, current_node, data["destination"]
                )
                data["start"] = current_node
                lats = [G.nodes[n]["y"] for n in new_route]
                lons = [G.nodes[n]["x"] for n in new_route]
                data["center"] = [
                    (min(lats) + max(lats)) / 2,
                    (min(lons) + max(lons)) / 2,
                ]
                data["old_congestion"] = old_cf
                data["new_congestion"] = new_cf
                data["time_saved"] = time_saved

                amb["step"] = 0
                amb["auto_drive"] = False
                amb["slider_override"] = True
                amb["reroute_count"] += 1
                amb["event_log"].append(
                    {
                        "event": "Accident Detected",
                        "node": current_node,
                        "old_congestion": old_cf,
                        "new_congestion": new_cf,
                        "time_saved": time_saved,
                    }
                )
                sync_ambulance_state(amb)
                st.success("Route rerouted around accident")
                st.rerun()
            except nx.NetworkXNoPath:
                st.error("No alternate path available")
        else:
            st.warning("Move the ambulance a few steps first before simulating an accident.")

    # Hospital path generation (only at pickup)
    can_gen_hospital = (
        amb.get("phase") == "ToPickup"
        and amb["step"] == len(route) - 1
        and amb["node"] == amb.get("pickup_node")
    )
    if st.button(
        "Generate Hospital Path",
        use_container_width=True,
        disabled=not can_gen_hospital,
    ):
        pickup = amb["pickup_node"]
        dists = {}
        for h in data["hospital_nodes"]:
            try:
                dists[h] = nx.shortest_path_length(G, pickup, h, weight="dynamic_weight")
            except nx.NetworkXNoPath:
                continue
        if dists:
            nearest_h = min(dists, key=dists.get)
            try:
                hosp_route = nx.shortest_path(G, pickup, nearest_h, weight="dynamic_weight")
                alts = build_alternate_routes(G, hosp_route, pickup, nearest_h)
                data["optimal_route"] = hosp_route
                data["start"] = pickup
                data["destination"] = nearest_h
                data["distances"] = dists
                data["alternate_routes"] = alts
                data["center"] = [
                    (G.nodes[pickup]["y"] + G.nodes[nearest_h]["y"]) / 2,
                    (G.nodes[pickup]["x"] + G.nodes[nearest_h]["x"]) / 2,
                ]
                amb["step"] = 0
                amb["phase"] = "ToHospital"
                amb["destination_hospital"] = nearest_h
                amb["event_log"].append({"event": "Patient Picked Up", "node": pickup})
                amb["auto_drive"] = False
                sync_ambulance_state(amb)
                st.success("Hospital route generated. Press Start.")
                st.rerun()
            except nx.NetworkXNoPath:
                st.error("No path to hospital from pickup.")
        else:
            st.error("No reachable hospitals found.")

    # Navigation buttons
    route = data["optimal_route"]  # refresh after possible reroute
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        start_dis = amb["status"] == "Arrived" and amb["phase"] == "ToHospital"
        if st.button("Start", use_container_width=True, disabled=start_dis):
            amb["auto_drive"] = True
            st.rerun()
    with col2:
        if st.button("Prev", disabled=amb["step"] == 0, use_container_width=True):
            amb["auto_drive"] = False
            amb["step"] -= 1
            sync_ambulance_state(amb)
            st.rerun()
    with col3:
        if st.button(
            "Next",
            disabled=amb["step"] >= len(route) - 1,
            use_container_width=True,
        ):
            amb["auto_drive"] = False
            amb["step"] += 1
            sync_ambulance_state(amb)
            st.rerun()
    with col4:
        if st.button(
            "End",
            disabled=amb["step"] >= len(route) - 1,
            use_container_width=True,
        ):
            amb["step"] = len(route) - 1
            sync_ambulance_state(amb)
            st.rerun()

    # Slider
    if amb.get("slider_override"):
        amb["slider_override"] = False

    slider_max = max(len(route) - 1, 1)
    slider_val = st.slider(
        "Position on Route",
        0,
        slider_max,
        min(amb["step"], slider_max),
        key=f"slider_{selected_ambulance}_{len(route)}",
    )
    # Only apply slider changes during manual control — never when auto-driving
    # or when the ambulance has already arrived (prevents reset-to-0 bug).
    if (
        not amb["auto_drive"]
        and not amb.get("slider_override")
        and amb["status"] != "Arrived"
        and slider_val != amb["step"]
    ):
        amb["step"] = slider_val
        sync_ambulance_state(amb)


    # ---- Reroute Event Log ----
    if amb["reroute_count"] > 0 or amb["event_log"]:
        st.markdown(
            '<div class="section-header"><h3>Event Log</h3></div>',
            unsafe_allow_html=True,
        )
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Total Reroutes", amb["reroute_count"])
        with mc2:
            if "old_congestion" in data and "new_congestion" in data and data["old_congestion"] > 0:
                pct = ((data["new_congestion"] - data["old_congestion"]) / data["old_congestion"]) * 100
                st.metric("Congestion Change", f"{pct:+.1f}%")
        with mc3:
            if "time_saved" in data:
                st.metric("Time Impact", f"{abs(data['time_saved']):.0f}s")

        if amb["event_log"]:
            with st.expander("Event Details", expanded=True):
                for idx, ev in enumerate(reversed(amb["event_log"])):
                    num = len(amb["event_log"]) - idx
                    if ev["event"] == "Patient Picked Up":
                        st.markdown(
                            f"**#{num}** — {ev['event']} at node `{ev['node']}`"
                        )
                    else:
                        saved_label = (
                            "saved" if ev.get("time_saved", 0) > 0 else "added"
                        )
                        st.markdown(
                            f"**#{num}** — {ev['event']} at node `{ev['node']}`  \n"
                            f"Congestion: {ev.get('old_congestion',0):.2f} → {ev.get('new_congestion',0):.2f} · "
                            f"Time: {abs(ev.get('time_saved',0)):.0f}s {saved_label}"
                        )
                    st.divider()

    # ---- Map ----
    route = data["optimal_route"]
    current_node = route[amb["step"]]
    lat, lon = G.nodes[current_node]["y"], G.nodes[current_node]["x"]

    m = folium.Map(location=data["center"], zoom_start=15, control_scale=True)

    # Alternate routes
    alt_colors = ["blue", "green", "orange", "purple"]
    for idx, alt in enumerate(data.get("alternate_routes", [])):
        if idx < len(alt_colors):
            pts = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in alt]
            folium.PolyLine(
                pts, color=alt_colors[idx], weight=4, opacity=0.6,
                tooltip=f"Alternate {idx+1}",
            ).add_to(m)

    # Old route (gray)
    if "old_route" in data:
        pts = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in data["old_route"]]
        folium.PolyLine(pts, color="gray", weight=3, opacity=0.4, tooltip="Previous Route").add_to(m)

    # Optimal route (red)
    opt_pts = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]
    folium.PolyLine(opt_pts, color="red", weight=5, opacity=0.85, tooltip="Optimal Route").add_to(m)

    # Start marker
    sn = data["start"]
    folium.Marker(
        location=(G.nodes[sn]["y"], G.nodes[sn]["x"]),
        popup="Start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)

    # Emergency / pickup marker
    if amb["phase"] == "ToPickup" and amb.get("pickup_node"):
        pn = amb["pickup_node"]
        folium.Marker(
            location=(G.nodes[pn]["y"], G.nodes[pn]["x"]),
            popup="Emergency Pickup",
            icon=folium.DivIcon(
                html=(
                    '<div style="width:18px;height:18px;background:#e74c3c;border-radius:50%;'
                    'border:3px solid #fff;box-shadow:0 2px 8px rgba(231,76,60,.5);'
                    'animation:blink 1s infinite;"></div>'
                    "<style>@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}</style>"
                ),
                icon_size=(18, 18),
                icon_anchor=(9, 9),
            ),
        ).add_to(m)

    # Hospital markers
    for h in data["hospital_nodes"]:
        is_dest = h == data.get("destination") and amb["phase"] == "ToHospital"
        folium.CircleMarker(
            location=(G.nodes[h]["y"], G.nodes[h]["x"]),
            radius=12 if is_dest else 8,
            color="red" if is_dest else "blue",
            fill=True,
            fillColor="#ff7f7f" if is_dest else "lightblue",
            fillOpacity=0.6,
            popup="Destination Hospital" if is_dest else "Hospital",
            weight=2,
        ).add_to(m)

    # Ambulance marker
    folium.Marker(
        location=(lat, lon),
        popup=f"{selected_ambulance} — Step {amb['step']+1}/{len(route)}",
        icon=folium.DivIcon(
            html=(
                '<div style="width:22px;height:22px;background:linear-gradient(135deg,#e74c3c,#c0392b);'
                'border-radius:50%;border:3px solid #fff;box-shadow:0 3px 10px rgba(231,76,60,.5);'
                'animation:pulse 2s infinite;"></div>'
                "<style>@keyframes pulse{0%{transform:scale(1);box-shadow:0 3px 10px rgba(231,76,60,.5)}"
                "50%{transform:scale(1.15);box-shadow:0 4px 16px rgba(231,76,60,.7)}"
                "100%{transform:scale(1);box-shadow:0 3px 10px rgba(231,76,60,.5)}}</style>"
            ),
            icon_size=(22, 22),
            icon_anchor=(11, 11),
        ),
    ).add_to(m)

    st_folium(m, height=650, use_container_width=True)

    # ---- Legend ----
    st.markdown(
        '<div class="section-header"><h3>Map Legend</h3></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:1rem;">'
        '<span class="legend-item"><span class="legend-dot" style="background:#e74c3c;"></span> Optimal Route</span>'
        '<span class="legend-item"><span class="legend-dot" style="background:#3498db;"></span> Alt Route 1</span>'
        '<span class="legend-item"><span class="legend-dot" style="background:#2ecc71;"></span> Alt Route 2</span>'
        '<span class="legend-item"><span class="legend-dot" style="background:#f39c12;"></span> Alt Route 3</span>'
        '<span class="legend-item"><span class="legend-dot" style="background:#2ecc71;border:2px solid #27ae60;"></span> Start</span>'
        '<span class="legend-item"><span class="legend-dot" style="background:#85c1e9;"></span> Hospital</span>'
        '<span class="legend-item"><span class="legend-dot" style="background:#e74c3c;border:2px solid #c0392b;"></span> Emergency</span>'
        + ('<span class="legend-item"><span class="legend-dot" style="background:#95a5a6;"></span> Previous Route</span>' if "old_route" in data else '')
        + '</div>',
        unsafe_allow_html=True,
    )

    # ---- Mission Status ----
    st.markdown(
        '<div class="section-header"><h3>Mission Status</h3></div>',
        unsafe_allow_html=True,
    )
    st.progress(progress, text=f"Mission Progress: {progress}%")

    if amb["step"] == 0:
        if amb["phase"] == "ToPickup":
            st.info("Dispatched to pickup location")
        elif amb["phase"] == "ToHospital":
            st.info("Transporting patient to hospital")
    elif amb["step"] >= len(route) - 1:
        if amb["phase"] == "ToPickup":
            st.warning("At pickup. Click Generate Hospital Path to continue.")
        elif amb["phase"] == "ToHospital":
            st.success("Arrived at hospital")
        else:
            st.success("Arrived at destination")

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("Step", f"{amb['step']+1}/{len(route)}")
    with sc2:
        st.metric("Progress", f"{progress}%")
    with sc3:
        total_dist = sum(
            _edge_attr(G, route[j], route[j + 1], "length", 0)
            for j in range(len(route) - 1)
        )
        st.metric("Distance", f"{total_dist/1000:.1f} km")
    with sc4:
        rem = max(len(route) - amb["step"] - 1, 0)
        if rem == 0:
            st.success("ARRIVED")
        else:
            st.info(f"{rem} steps left")

# ================================================================
# AUTO-DRIVE RERUN LOOP
# ================================================================
if any_auto_drive:
    sleep_for = max(0.0, MOVE_INTERVAL_SEC - (time.time() - st.session_state.last_move_ts))
    if sleep_for > 0:
        time.sleep(sleep_for)
    st.rerun()
