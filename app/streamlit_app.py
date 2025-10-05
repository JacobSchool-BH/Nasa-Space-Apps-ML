# app/streamlit_app.py — Exoplanet Explorer (Binary: CONFIRMED vs FALSE POSITIVE)
# Tabs: Novice (3D viz with model predictions), Researcher (Train), Tester (Batch classify)
# Robust CSV ingestion, engineered features, NO label leakage, and schema-stable prediction.

import os, io, json, time, traceback, csv
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Optional TensorFlow (needed for training/prediction)
try:
    import tensorflow as tf
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    except Exception:
        pass
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
except Exception:
    tf = None

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import plotly.graph_objects as go

# ---------- Page ----------
st.set_page_config(page_title="Exoplanet Explorer (Binary)", page_icon="🪐", layout="wide")

# ---------- Paths / constants ----------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RAW_DIR    = os.path.join(ROOT_DIR, "data", "raw")
UPLOAD_DIR = os.path.join(ROOT_DIR, "data", "user_uploads")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

LABEL_NAMES = ["FALSE POSITIVE", "CONFIRMED"]  # binary order: 0, 1
DISP_COLOR  = {"CONFIRMED":"#4ade80", "CANDIDATE":"#60a5fa", "FALSE POSITIVE":"#f43f5e"}
PRED_COLOR  = {"CONFIRMED":"#4ade80", "FALSE POSITIVE":"#f43f5e"}

NUMERIC_FEATURES = ["period","duration","depth","prad","impact","snr","steff","slogg","srad"]

K2_FILE_CANDIDATES  = ["k2pandc.csv", "k2pandac.csv", "k2_planets_candidates.csv"]
TOI_FILE_CANDIDATES = ["toi.csv", "toi_catalog.csv", "toi_release.csv"]

ORDER_FEATURES = {
    "Orbital period (days) — inner = shorter": {"field":"period","desc":"Inner rings show short-period planets; outer rings long-period."},
    "Planet radius (Earth radii) — inner = smaller": {"field":"prad","desc":"Inner rings are small planets; outer rings larger ones."},
    "Signal-to-noise ratio — inner = higher": {"field":"snr","desc":"Inner rings are high-SNR (clearer signals)."},
    "Transit depth (ppm) — inner = deeper": {"field":"depth","desc":"Deeper transits are inside."},
    "Stellar Teff (K) — inner = cooler host": {"field":"steff","desc":"Cooler hosts inside; hotter hosts outside."},
}

# ---------- IO helpers ----------
def _first_existing(dirpath, names):
    for n in names:
        p = os.path.join(dirpath, n)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None

def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, engine="python", sep=None, on_bad_lines="skip")
    except Exception:
        pass
    for enc in ("utf-8","latin-1"):
        try:
            return pd.read_csv(path, engine="python", sep=None, encoding=enc, on_bad_lines="skip")
        except Exception:
            for sep in (",",";","\t","|"):
                try:
                    return pd.read_csv(path, engine="python", sep=sep, encoding=enc, on_bad_lines="skip")
                except Exception:
                    continue
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if "<html" in text.lower(): return pd.DataFrame()
        from io import StringIO
        return pd.read_csv(StringIO(text), engine="python", sep=None, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()

def safe_read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    text = None
    for enc in ("utf-8","utf-16","latin-1"):
        try:
            text = file_bytes.decode(enc); break
        except Exception:
            continue
    if text is None: text = file_bytes.decode("utf-8", errors="ignore")
    try:
        sample = text[:5000]
        dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"])
        sep = dialect.delimiter
    except Exception:
        sep = None
    from io import StringIO
    try:
        return pd.read_csv(StringIO(text), engine="python", sep=sep if sep else None, on_bad_lines="skip")
    except Exception:
        for sep in (",",";","\t","|"):
            try:
                return pd.read_csv(StringIO(text), engine="python", sep=sep, on_bad_lines="skip")
            except Exception:
                continue
    return pd.DataFrame()

# ---------- Header normalization / features ----------
def _coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _clean_unified(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    df = df.copy()
    koi_map = {
        "koi_period":"period","koi_duration":"duration","koi_depth":"depth","koi_prad":"prad","koi_impact":"impact",
        "koi_snr":"snr","koi_model_snr":"snr","koi_steff":"steff","koi_slogg":"slogg","koi_srad":"srad","koi_disposition":"disposition"
    }
    for src, dst in koi_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    _coerce_numeric(df, NUMERIC_FEATURES)
    if "mission" not in df.columns:
        df["mission"] = "kepler"
    return df

def normalize_catalog_headers(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    orig = list(df.columns)
    lowmap = {}
    for c in orig:
        lc = c.strip().lower()
        lc = lc.replace("(days)","").replace("(hours)","").replace("(ppm)","") \
               .replace("(r_earth)","").replace("(r_sun)","").replace("(k)","") \
               .replace("(cgs)","").strip()
        lowmap[lc] = c
    alias = {
        "period":"period","orbital_period":"period","pl_orbper":"period","toi_period":"period",
        "duration":"duration","t14":"duration","transit_duration":"duration","toi_duration":"duration",
        "depth":"depth","tran_depth":"depth","transit_depth":"depth","toi_depth":"depth",
        "planet_radius":"prad","pl_rade":"prad","radius_re":"prad","prad":"prad",
        "impact":"impact","b":"impact",
        "snr":"snr","model_snr":"snr","koi_snr":"snr",
        "st_teff":"steff","teff":"steff","stellar_teff":"steff","koi_steff":"steff",
        "st_logg":"slogg","logg":"slogg","koi_slogg":"slogg",
        "st_rad":"srad","stellar_radius":"srad","koi_srad":"srad",
        "disposition":"disposition","k2_disposition":"disposition","koi_disposition":"disposition","tfopwg_disp":"disposition",
        "status":"disposition","label":"disposition",
        "mission":"mission",
    }
    remap = {}
    for lc, origc in lowmap.items():
        if lc in alias:
            remap[origc] = alias[lc]
    out = df.rename(columns=remap).copy()
    if "mission" not in out.columns:
        if any(k in lowmap for k in ("koi_period","koi_prad","koi_disposition")):
            out["mission"] = "kepler"
        elif any(k in lowmap for k in ("pl_orbper","t14","k2_disposition")):
            out["mission"] = "k2"
        elif any(k in lowmap for k in ("toi_period","tfopwg_disp")):
            out["mission"] = "tess"
        else:
            out["mission"] = "unknown"
    out = _clean_unified(out)
    return out

EXTRA_NUMERIC_CANDIDATES = [
    "koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_co","koi_fpflag_ec",
    "koi_depth_err1","koi_depth_err2","koi_duration_err1","koi_duration_err2",
    "koi_prad_err1","koi_prad_err2","koi_period_err1","koi_period_err2",
    "koi_steff_err1","koi_steff_err2","koi_slogg_err1","koi_slogg_err2",
    "koi_srad_err1","koi_srad_err2",
    "koi_kepmag","koi_gmag","koi_rmag","koi_imag","koi_zmag","koi_jmag","koi_hmag","koi_kmag",
    "koi_max_mult_ev"
]

def build_features(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Normalize headers, create engineered features, and impute.
    Returns (df, feature_cols_without_mission_onehot). 'mission' remains as a separate column.
    """
    df = normalize_catalog_headers(df_raw)
    for c in NUMERIC_FEATURES:
        if c not in df.columns: df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    extras = [c for c in EXTRA_NUMERIC_CANDIDATES if c in df.columns]
    for c in extras:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for f in ["koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_co","koi_fpflag_ec"]:
        if f in df.columns:
            df[f] = df[f].fillna(0.0).clip(0, 1)

    for b in ["period","duration","depth","snr","prad","srad","steff"]:
        df[f"log_{b}"] = np.log1p(df[b].clip(lower=0))

    df["depth_per_pr2"]   = df["depth"] / (df["prad"].replace(0,np.nan)**2)
    df["dur_over_period"] = df["duration"] / df["period"].replace(0,np.nan)
    df["prad_over_srad"]  = df["prad"] / df["srad"].replace(0,np.nan)
    df["teff_over_logg"]  = df["steff"] / df["slogg"].replace(0,np.nan)
    df.replace([np.inf,-np.inf], np.nan, inplace=True)

    base_feats = NUMERIC_FEATURES + \
        [f"log_{b}" for b in ["period","duration","depth","snr","prad","srad","steff"]] + \
        ["depth_per_pr2","dur_over_period","prad_over_srad","teff_over_logg"] + \
        extras
    feat_cols = [c for c in base_feats if c in df.columns and not any(t in c.lower() for t in ["disposition","label","ground_truth"])]

    for c in feat_cols:
        df[c] = df[c].fillna(df[c].median())

    if "mission" not in df.columns:
        df["mission"] = "unknown"

    return df, feat_cols

# ---------- Model I/O ----------
@st.cache_resource
def load_model_bundle():
    if tf is None: return None
    try:
        model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "exoplanet_mlp.keras"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
        meta_path = os.path.join(MODELS_DIR, "meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        return model, scaler, meta
    except Exception:
        return None

def build_mlp(input_dim: int, n_classes: int = 2, hidden=[192,96], dropout=0.25, lr=5e-4):
    if tf is None:
        st.error("TensorFlow not installed. Run: pip install tensorflow (or tensorflow-macos)"); raise RuntimeError
    from tensorflow.keras import layers, optimizers, Model, Input
    inputs = Input(shape=(input_dim,), name="features")
    x = inputs
    for h in hidden:
        x = layers.Dense(h, activation="relu", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="probs")(x)
    model = Model(inputs, outputs, name="exoplanet_mlp_binary")
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )
    return model

# ---------- Manual one-hot using saved mission categories ----------
def mission_ohe_matrix(missions, mission_cats):
    """
    Robust, order-stable one-hot for 'mission'.
    missions: array-like (N,) of strings
    mission_cats: list of categories in the exact order the model expects
    """
    # Normalize inputs
    if mission_cats is None:
        mission_cats = []
    elif isinstance(mission_cats, (np.ndarray, tuple, set)):
        mission_cats = list(mission_cats)
    mission_cats = [str(x) for x in mission_cats]

    # Missions -> 1-D ndarray[str]
    missions = pd.Series(missions).astype(str).fillna("unknown").to_numpy()
    N = missions.shape[0]
    K = len(mission_cats)
    if K == 0:
        return np.zeros((N, 0), dtype=np.float32)

    idx = {m: i for i, m in enumerate(mission_cats)}
    mat = np.zeros((N, K), dtype=np.float32)
    for r, m in enumerate(missions):
        j = idx.get(m)
        if j is not None:
            mat[r, j] = 1.0
    return mat


def predict_if_possible(df_input):
    """
    Schema-stable predictor:
      - Accepts dict / Series / DataFrame
      - Rebuilds engineered features
      - Ensures scaler gets a 2-D float array (N, n_features)
      - Aligns feature count to meta['feature_names']
    """
    bundle = load_model_bundle()
    if bundle is None:
        return None, None
    model, scaler, meta = bundle

    # ---- meta normalization ----
    if not isinstance(meta, dict):
        raise RuntimeError("Invalid meta.json (not a dict). Retrain to re-create artifacts.")
    feat_names = meta.get("feature_names")
    if isinstance(feat_names, np.ndarray):
        feat_names = feat_names.tolist()
    elif isinstance(feat_names, tuple):
        feat_names = list(feat_names)
    elif isinstance(feat_names, str):
        # allow JSON string or comma list
        try:
            tmp = json.loads(feat_names)
            feat_names = tmp if isinstance(tmp, list) else [feat_names]
        except Exception:
            feat_names = [s.strip() for s in feat_names.split(",") if s.strip()]
    if not isinstance(feat_names, list) or not feat_names:
        raise RuntimeError("meta['feature_names'] missing or invalid.")
    feat_names = [str(c) for c in feat_names]

    mission_cats = meta.get("mission_categories", [])
    if isinstance(mission_cats, np.ndarray):
        mission_cats = mission_cats.tolist()
    elif isinstance(mission_cats, tuple):
        mission_cats = list(mission_cats)
    elif isinstance(mission_cats, str):
        try:
            tmp = json.loads(mission_cats)
            mission_cats = tmp if isinstance(tmp, list) else [mission_cats]
        except Exception:
            mission_cats = [s.strip() for s in mission_cats.split(",") if s.strip()]
    mission_cats = [str(x) for x in mission_cats]

    # Split names by type (numeric/engineered vs mission one-hot)
    mission_cols = [c for c in feat_names if c.startswith("mission_")]
    num_feat_names = [c for c in feat_names if c not in mission_cols]

    # ---- coerce input to DataFrame ----
    if isinstance(df_input, pd.Series):
        df_input = pd.DataFrame([df_input.to_dict()])
    elif isinstance(df_input, dict):
        # dict of scalars -> single-row
        df_input = pd.DataFrame([df_input])
    elif not isinstance(df_input, pd.DataFrame):
        try:
            df_input = pd.DataFrame(df_input)
        except Exception:
            raise RuntimeError("Input must be DataFrame/Series/dict/array-like.")

    if df_input is None or df_input.empty:
        raise RuntimeError("No rows to predict.")

    # ---- engineered features ----
    df_fe, _ = build_features(df_input.copy())

    # Ensure 'mission' column exists and is string
    if "mission" not in df_fe.columns:
        df_fe["mission"] = "unknown"
    df_fe["mission"] = df_fe["mission"].astype(str)

    # Ensure all numeric/engineered columns exist and are numeric
    for c in num_feat_names:
        if c not in df_fe.columns:
            df_fe[c] = np.nan
        df_fe[c] = pd.to_numeric(df_fe[c], errors="coerce")
        med = df_fe[c].median()
        if pd.isna(med): med = 0.0
        df_fe[c] = df_fe[c].fillna(med)

    # ---- build X (N, n_features_expected) ----
    # Numeric block first, strictly ordered and float32
    X_num = df_fe.loc[:, [str(c) for c in num_feat_names]].to_numpy(dtype=np.float32)

    # Manual mission one-hot next
    X_cat = mission_ohe_matrix(df_fe["mission"].to_numpy(), mission_cats).astype(np.float32)

    X = np.hstack([X_num, X_cat]) if X_cat.shape[1] > 0 else X_num

    # Align to expected feature length
    expected = len(feat_names)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != expected:
        if X.shape[1] < expected:
            pad = np.zeros((X.shape[0], expected - X.shape[1]), dtype=X.dtype)
            X = np.hstack([X, pad])
        else:
            X = X[:, :expected]

    # ---- scale + predict (defensive casting to avoid list/tuple/1D errors) ----
    try:
        Xs = scaler.transform(np.asarray(X, dtype=np.float32))
    except Exception:
        # one more attempt with float64
        Xs = scaler.transform(np.asarray(X, dtype=np.float64))

    probs = model.predict(Xs, verbose=0)

    # normalize output to 2 columns (FP, CONF)
    if probs.ndim == 1:
        probs = np.stack([1.0 - probs, probs], axis=1)
    elif probs.shape[1] != 2:
        if probs.shape[1] > 2:
            probs = probs[:, -2:]
        else:
            # pad to 2 cols if needed
            pad = np.zeros((probs.shape[0], 2 - probs.shape[1]), dtype=probs.dtype)
            probs = np.hstack([probs, pad])

    idx = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    label_names = meta.get("label_names", ["FALSE POSITIVE", "CONFIRMED"])
    if not isinstance(label_names, list) or len(label_names) != 2:
        label_names = ["FALSE POSITIVE", "CONFIRMED"]
    preds = [label_names[int(i)] for i in idx]
    return preds, conf

# ---------- Viz data ----------
def _load_koi_remote():
    # Lightweight subset via TAP: change columns as you like
    url = (
      "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
      "?query=select+kepoi_name,koi_disposition,koi_period,koi_duration,koi_depth,"
      "koi_prad,koi_impact,koi_snr,koi_steff,koi_slogg,koi_srad+from+kepler_koi"
      "&format=csv"
    )
    d = pd.read_csv(url)
    d = d.rename(columns={
        "kepoi_name":"name",
        "koi_disposition":"disposition",
        "koi_period":"period",
        "koi_duration":"duration",
        "koi_depth":"depth",
        "koi_prad":"prad",
        "koi_impact":"impact",
        "koi_snr":"snr",
        "koi_steff":"steff",
        "koi_slogg":"slogg",
        "koi_srad":"srad",
    })
    d["mission"] = "kepler"
    return d

@st.cache_data(show_spinner=False)
def load_union_galaxy():
    frames = []
    koi_path = os.path.join(RAW_DIR, "koi_cumulative.csv")
    if os.path.exists(koi_path):
        d = safe_read_csv(koi_path)
        d = normalize_catalog_headers(d); d["mission"] = "kepler"
        frames.append(d)
    else:
        try:
            frames.append(_load_koi_remote())
        except Exception:
            pass

    # (You can add similar remote fallbacks for K2/TOI later.)

    if not frames:
        st.error("No catalog found locally and remote fetch failed. Upload a CSV or commit a sample to the repo.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ---------- Plotly renderers ----------
def render_galaxy3d_plotly(planets, bg="#0b1020", key=None):
    if not planets:
        st.info("No planets to display."); return
    xs=[p["x"] for p in planets]; ys=[p["y"] for p in planets]; zs=[p["z"] for p in planets]
    clr=[p.get("color","#60a5fa") for p in planets]
    txt=[p.get("label_text", f'{p["name"]} • {p.get("disposition","")}') for p in planets]
    siz=[max(2, float(p.get("r",1.0))*3) for p in planets]

    fig = go.Figure(data=[
        go.Scatter3d(x=xs,y=ys,z=zs,mode="markers",
            marker=dict(size=[s*1.8 for s in siz], color=clr, opacity=0.18, sizemode="diameter"),
            hoverinfo="skip", showlegend=False),
        go.Scatter3d(x=xs,y=ys,z=zs,mode="markers",
            marker=dict(size=siz, color=clr, opacity=0.95, sizemode="diameter",
                        line=dict(width=1, color="rgba(255,255,255,0.35)")),
            text=txt, hovertemplate="%{text}<extra></extra>", showlegend=False)
    ])
    fig.update_layout(height=720, scene=dict(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        bgcolor=bg), paper_bgcolor=bg, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True, theme=None, key=key)

def render_solar_style_plotly(planets, bg="#000", rings=8, feature_for_order="period", key=None):
    if not planets:
        st.info("No planets to display."); return
    vals=[]
    for i,p in enumerate(planets):
        v=p.get(feature_for_order,None)
        try: v=float(v) if v is not None else None
        except: v=None
        vals.append((1e99 if v is None else v, i))
    vals.sort(key=lambda t: t[0])

    outer_r, inner_r = 18.0, 3.0
    ring_radii = np.linspace(inner_r, outer_r, rings)
    ring_members=[[] for _ in range(rings)]
    for k, (_, idx) in enumerate(vals):
        ring_members[k % rings].append(idx)

    u=np.linspace(0,2*np.pi,30); v=np.linspace(0,np.pi,20); R=1.2
    xs=R*np.outer(np.cos(u), np.sin(v)); ys=R*np.outer(np.sin(u), np.sin(v)); zs=R*np.outer(np.ones_like(u), np.cos(v))
    data=[
        go.Surface(x=xs,y=ys,z=zs,showscale=False,opacity=0.9,
                   colorscale=[[0,"#fff1a6"],[0.5,"#ffd85b"],[1.0,"#ffb300"]],
                   hoverinfo="skip"),
        go.Scatter3d(x=[0],y=[0],z=[0],mode="markers",
                     marker=dict(size=12, color="#ffd85b", opacity=1.0),
                     text=["Central Star"], hovertemplate="%{text}<extra></extra>", showlegend=False)
    ]
    for ri, rad in enumerate(ring_radii):
        theta=np.linspace(0,2*np.pi,240)
        tilt_x=(ri%3)*np.deg2rad(4.0); tilt_y=((ri+1)%3)*np.deg2rad(3.0)
        x=rad*np.cos(theta); y=rad*np.sin(theta); z=np.zeros_like(theta)
        y2=y*np.cos(tilt_x)-z*np.sin(tilt_x); z2=y*np.sin(tilt_x)+z*np.cos(tilt_x)
        x3=x*np.cos(tilt_y)+z2*np.sin(tilt_y); z3=-x*np.sin(tilt_y)+z2*np.cos(tilt_y)
        data.append(go.Scatter3d(x=x3,y=y2,z=z3,mode="lines",
                                 line=dict(width=1,color="rgba(255,255,255,0.15)"),
                                 hoverinfo="skip", showlegend=False))
    px,py,pz,pc,pt,ps=[],[],[],[],[],[]
    for ri, members in enumerate(ring_members):
        if not members: continue
        rad=ring_radii[ri]; n=len(members)
        tilt_x=(ri%3)*np.deg2rad(4.0); tilt_y=((ri+1)%3)*np.deg2rad(3.0)
        for j, idx in enumerate(members):
            ang=2*np.pi*(j/n if n>0 else 0)
            jr=np.random.uniform(-0.25,0.25); jz=np.random.uniform(-0.2,0.2)
            x=(rad+jr)*np.cos(ang); y=(rad+jr)*np.sin(ang); z=jz
            y2=y*np.cos(tilt_x)-z*np.sin(tilt_x); z2=y*np.sin(tilt_x)+z*np.cos(tilt_x)
            x3=x*np.cos(tilt_y)+z2*np.sin(tilt_y); z3=-x*np.sin(tilt_y)+z2*np.cos(tilt_y)
            p=planets[idx]
            px.append(x3); py.append(y2); pz.append(z3)
            pc.append(p.get("color","#60a5fa"))
            pt.append(p.get("label_text", f'{p.get("name","")} • {p.get("disposition","")}'))
            pr=float(p.get("r",1.0)); ps.append(max(3, min(14, pr*3.2)))
    data += [
        go.Scatter3d(x=px,y=py,z=pz,mode="markers",
                     marker=dict(size=[min(18,s*1.8) for s in ps], color=pc, opacity=0.18, sizemode="diameter"),
                     hoverinfo="skip", showlegend=False),
        go.Scatter3d(x=px,y=py,z=pz,mode="markers",
                     marker=dict(size=ps, color=pc, opacity=0.98, sizemode="diameter",
                                 line=dict(width=1,color="rgba(255,255,255,0.35)")),
                     text=pt, hovertemplate="%{text}<extra></extra>", showlegend=False)
    ]
    fig = go.Figure(data=data)
    fig.update_layout(height=720, scene=dict(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        bgcolor=bg, aspectmode="data"),
        paper_bgcolor=bg, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True, theme=None, key=key)

# ---------- Batch classify ----------
def batch_classify_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: raise ValueError("Empty dataframe.")
    dfu, _ = build_features(df)  # normalize + engineer
    preds, confs = predict_if_possible(dfu)
    if preds is None:
        raise RuntimeError("No trained model found in /models. Train in the Researcher tab first.")
    out = normalize_catalog_headers(df).copy()
    out["prediction"] = preds
    out["confidence"] = confs
    return out

# ---------- Model status ----------
def model_status():
    ok = False; msg = "No model loaded"
    try:
        bundle = load_model_bundle()
        if bundle is None:
            msg = "Model bundle not found in /models"
        else:
            model, scaler, meta = bundle
            lbls = (meta or {}).get("label_names", LABEL_NAMES)
            ok = True
            msg = f"Loaded: {getattr(model,'name','model')} • labels={lbls}"
    except Exception as e:
        msg = f"Load error: {e}"
    with st.sidebar.expander("🧠 Model Status", expanded=False):
        st.write(msg)
        if not ok:
            st.caption("Train in Researcher → “Train Binary Model”.")
    return ok, msg

# ---------- UI ----------
def main():
    st.title("🪐 Exoplanet Explorer — Binary (CONFIRMED vs FALSE POSITIVE)")
    ok_model, _ = model_status()

    mode = st.sidebar.radio(
        "Choose interface",
        ["Novice • 3D Galaxy", "Researcher • Train (Binary)", "Tester • Batch Classifier"],
        key="mode_radio"
    )

    # ===== Novice =====
    if mode.startswith("Novice"):
        st.subheader("✨ 3D Galaxy of Exoplanets — model predictions on hover (Candidates colored by model)")

        catalog = load_union_galaxy()
        if catalog is None or catalog.empty:
            st.error("No catalog data found. Put CSVs in data/raw: koi_cumulative.csv, k2pandac.csv (or k2pandc.csv), toi.csv")
            st.stop()

        with st.expander("Filters", expanded=True):
            c1, c2, c3 = st.columns(3)
            missions = c1.multiselect(
                "Mission(s)", sorted(catalog["mission"].dropna().astype(str).unique().tolist()),
                default=sorted(catalog["mission"].dropna().astype(str).unique().tolist()),
                key="novice_missions"
            )
            period_max_i = int(max(1, np.ceil(float(catalog["period"].max()))))
            period_range = c1.slider("Orbital Period (days)", 0, period_max_i, (0, min(500, period_max_i)), step=1, key="novice_period")
            prad_max = float(catalog["prad"].max()); prad_hi_default = min(5.0, max(1.0, prad_max))
            prad_range = c2.slider("Planet Radius (Earth radii)", 0.0, max(1.0, prad_max), (0.0, prad_hi_default), step=0.1, key="novice_prad")
            snr_max_i = int(np.ceil(float(catalog["snr"].max()))) if "snr" in catalog.columns else 1
            snr_min_val = c2.slider("Min SNR", 0, max(1, snr_max_i), min(10, max(0, snr_max_i)), step=1, key="novice_snr")
            teff_min_i = int(50 * np.floor(float(catalog["steff"].min())/50.0))
            teff_max_i = int(50 * np.ceil(float(catalog["steff"].max())/50.0))
            teff_range = c3.slider("Stellar Teff (K)", teff_min_i, teff_max_i, (teff_min_i, min(7000, teff_max_i)), step=50, key="novice_teff")
            pretty_mode = c3.toggle("Solar-style layout (central star)", value=True, key="novice_pretty_toggle")
            order_label = c3.selectbox("Solar layout: order rings by…", list(ORDER_FEATURES.keys()), index=0, key="novice_order_feature",
                                       help="Controls what 'closeness to the star' means.")
            max_n = c2.slider("Max planets to render", 100, 5000, 1200, 100, key="novice_maxn")

        f = catalog[
            catalog["mission"].isin(missions) &
            catalog["period"].between(period_range[0], period_range[1]) &
            catalog["prad"].between(prad_range[0], prad_range[1]) &
            (catalog["snr"] >= snr_min_val) &
            catalog["steff"].between(teff_range[0], teff_range[1])
        ].copy()

        st.write(f"Showing **{len(f):,}** of **{len(catalog):,}** objects.")
        rows = f.head(max_n).reset_index(drop=True)

        # Color: candidates take model color; others keep catalog color
        def _pick_color(row):
            disp = str(row.get("disposition","")).upper()
            pred = str(row.get("predicted",""))
            if disp == "CANDIDATE" and pred in PRED_COLOR:
                return PRED_COLOR[pred]
            return DISP_COLOR.get(disp, "#60a5fa")
        rows["display_color"] = rows.apply(_pick_color, axis=1)

        def _get_float(row, col):
            return None if col not in row or pd.isna(row[col]) else float(row[col])

        planets_payload=[]
        for i, row in rows.reset_index(drop=True).iterrows():
            planets_payload.append(dict(
                name=str(row["name"]) if "name" in rows.columns else f"OBJ-{i}",
                x=float(row["x"]), y=float(row["y"]), z=float(row["z"]),
                color=row["display_color"],
                r=float(max(0.6, row["prad"] if "prad" in rows.columns and not pd.isna(row["prad"]) else 1.0)),
                mission=str(row["mission"] if "mission" in rows.columns else "unknown"),
                disposition=str(row["disposition"]),
                period=_get_float(row, "period"),
                prad=_get_float(row, "prad"),
                snr=_get_float(row, "snr"),
                depth=_get_float(row, "depth"),
                steff=_get_float(row, "steff"),
            ))

        chart_key = f"viz_{'solar' if pretty_mode else 'pca'}"
        if pretty_mode:
            render_solar_style_plotly(planets_payload, bg="#000000",
                                      rings=8, feature_for_order=ORDER_FEATURES[order_label]["field"], key=chart_key)
        else:
            render_galaxy3d_plotly(planets_payload, bg="#0b1020", key=chart_key)

    # ===== Researcher (Binary Train) =====
    elif mode.startswith("Researcher"):
        st.subheader("📦 Train Binary Classifier (CONFIRMED vs FALSE POSITIVE)")
        st.caption("KOI/K2/TOI + optional upload. Rows labeled 'CANDIDATE' are excluded from training. No label leakage.")

        up = st.file_uploader("Upload CSV (KOI/K2/TOI style or your own columns)", type=["csv"], key="researcher_upload")
        df_upload = None
        if up is not None:
            try:
                raw = up.read()
                df_upload = safe_read_csv_bytes(raw)
                if df_upload is not None and not df_upload.empty:
                    with st.expander(f"🔎 Raw upload: {up.name}", expanded=False):
                        st.dataframe(df_upload.head(15), use_container_width=True)
                    df_upload = normalize_catalog_headers(df_upload)
                    with st.expander("Normalized upload", expanded=False):
                        st.dataframe(df_upload.head(15), use_container_width=True)
            except Exception as e:
                st.error("Could not read/normalize the uploaded CSV."); st.exception(e)

        st.divider()
        st.write("### Sources & Hyperparameters")
        c1, c2 = st.columns(2)
        use_koi = c1.checkbox("Include KOI (data/raw/koi_cumulative.csv)", value=os.path.exists(os.path.join(RAW_DIR, "koi_cumulative.csv")), key="bin_koi")
        use_k2  = c1.checkbox("Include K2  (k2pandac.csv / k2pandc.csv)", value=False, key="bin_k2")
        use_toi = c1.checkbox("Include TOI (toi.csv)", value=False, key="bin_toi")
        include_upload = c1.checkbox("Include uploaded CSV", value=(df_upload is not None), key="bin_upload")
        turbo = c1.toggle("Turbo mode (downsample large datasets)", value=True, key="bin_turbo")

        lr     = c2.number_input("Learning rate", value=5e-4, format="%.5f", key="bin_lr")
        epochs = c2.number_input("Max epochs", min_value=5, value=80, step=1, key="bin_epochs")
        batch  = c2.number_input("Batch size", min_value=64, value=256, step=64, key="bin_batch")
        drop   = c2.slider("Dropout", 0.0, 0.6, 0.25, 0.05, key="bin_dropout")
        h1     = c2.number_input("Hidden layer 1", min_value=16, value=192, step=16, key="bin_h1")
        h2     = c2.number_input("Hidden layer 2", min_value=0, value=96, step=16, key="bin_h2")
        patience = c2.number_input("EarlyStopping patience", 3, 30, 8, 1, key="bin_patience")

        if st.button("🚀 Train Binary Model", type="primary", key="btn_train_binary"):
            if tf is None:
                st.error("TensorFlow not installed. Run: pip install tensorflow (or tensorflow-macos)"); st.stop()

            with st.spinner("Training binary model… (avoid reloading the page)"):
                frames = []; sources = []

                koi_path = os.path.join(RAW_DIR, "koi_cumulative.csv")
                if use_koi and os.path.exists(koi_path):
                    d = safe_read_csv(koi_path)
                    d = normalize_catalog_headers(d); d["mission"]="kepler"
                    frames.append(d); sources.append("koi")

                k2p = _first_existing(RAW_DIR, K2_FILE_CANDIDATES)
                if use_k2 and k2p:
                    d = safe_read_csv(k2p)
                    if d is not None and not d.empty:
                        d = normalize_catalog_headers(d); d["mission"]="k2"
                        frames.append(d); sources.append(f"k2:{os.path.basename(k2p)}")

                toip = _first_existing(RAW_DIR, TOI_FILE_CANDIDATES)
                if use_toi and toip:
                    d = safe_read_csv(toip)
                    if d is not None and not d.empty:
                        d = normalize_catalog_headers(d); d["mission"]="tess"
                        frames.append(d); sources.append(f"toi:{os.path.basename(toip)}")

                if include_upload and df_upload is not None and not df_upload.empty:
                    frames.append(df_upload.copy()); sources.append("upload")

                if not frames:
                    st.error("No data sources selected or found."); st.stop()

                data_raw = pd.concat(frames, ignore_index=True, sort=False)
                df_fe, feat_cols = build_features(data_raw)

                if "disposition" not in df_fe.columns:
                    st.error("No disposition/label column found after normalization. Ensure your CSV has one."); st.stop()

                disp = df_fe["disposition"].astype(str).str.upper()
                disp_bin = disp.map(lambda s: "CONFIRMED" if s=="CONFIRMED" else ("FALSE POSITIVE" if s in ["FALSE POSITIVE","FP"] else "CANDIDATE"))
                mask = disp_bin.isin(LABEL_NAMES)  # keep only FP / CONFIRMED
                if not mask.any():
                    st.error("After filtering, no rows labeled CONFIRMED or FALSE POSITIVE were found."); st.stop()

                df_bin = df_fe[mask].copy()
                df_bin["disposition"] = disp_bin[mask]
                y = df_bin["disposition"].map({"FALSE POSITIVE":0, "CONFIRMED":1}).astype(int).values

                # Manual mission categories (stable schema saved to meta)
                mission_cats = sorted(df_bin["mission"].astype(str).unique().tolist())

                X_num = df_bin[feat_cols].astype("float32").to_numpy()
                X_cat = mission_ohe_matrix(df_bin["mission"].astype(str).to_numpy(), mission_cats)
                X = np.hstack([X_num, X_cat]) if X_cat.shape[1] > 0 else X_num

                if turbo and len(df_bin) > 30000:
                    st.info("Turbo mode: downsampling to ~30k rows for faster training.")
                    rng = np.random.RandomState(42)
                    parts=[]
                    for lbl in [0,1]:
                        idxs = np.where(y==lbl)[0]
                        n = min(len(idxs), 15000)
                        keep = rng.choice(idxs, size=n, replace=False)
                        parts.append(keep)
                    keep_idx = np.concatenate(parts)
                    X = X[keep_idx]; y = y[keep_idx]

                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
                X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=42)

                scaler = StandardScaler().fit(X_tr)
                X_tr_s = scaler.transform(X_tr).astype("float32")
                X_va_s = scaler.transform(X_va).astype("float32")
                X_te_s = scaler.transform(X_te).astype("float32")

                classes = np.unique(y_tr)
                cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
                class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

                hidden = [int(h1)] + ([int(h2)] if h2>0 else [])
                model = build_mlp(input_dim=X_tr_s.shape[1], n_classes=2, hidden=hidden, dropout=float(drop), lr=float(lr))
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(patience), restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, int(patience/2)), verbose=0),
                ]

                model.fit(
                    X_tr_s, y_tr,
                    validation_data=(X_va_s, y_va),
                    epochs=int(epochs), batch_size=int(batch),
                    class_weight=class_weight, callbacks=callbacks, verbose=0
                )

                probs = model.predict(X_te_s, verbose=0)
                if probs.ndim == 1:
                    probs = np.stack([1.0-probs, probs], axis=1)
                y_pred = probs.argmax(axis=1)
                rep = classification_report(y_te, y_pred, target_names=LABEL_NAMES, output_dict=True, digits=4)
                cm  = confusion_matrix(y_te, y_pred, labels=[0,1])
                try:
                    auc = roc_auc_score(y_te, probs[:,1])
                except Exception:
                    auc = None
                acc = float(rep["accuracy"])

                # Save artifacts: model, scaler, and meta with full schema
                model.save(os.path.join(MODELS_DIR, "exoplanet_mlp.keras"))
                joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
                with open(os.path.join(MODELS_DIR, "meta.json"), "w") as f:
                    json.dump({
                        "feature_names": feat_cols + [f"mission_{c}" for c in mission_cats],  # list of strings
                        "mission_categories": mission_cats,                                     # list of strings
                        "training": "binary_all_sources_manual_ohe",
                        "label_names": ["FALSE POSITIVE","CONFIRMED"],
                        "sources": sources
                    }, f, indent=2)


                # Clean up any legacy encoder files to avoid confusion
                enc_path = os.path.join(MODELS_DIR, "encoders.joblib")
                if os.path.exists(enc_path):
                    try: os.remove(enc_path)
                    except Exception: pass
                thr_path = os.path.join(MODELS_DIR, "thresholds.json")
                if os.path.exists(thr_path):
                    try: os.remove(thr_path)
                    except Exception: pass

                st.success(f"Binary model trained. Test accuracy: {acc:.4f}" + (f" • ROC-AUC: {auc:.4f}" if auc is not None else ""))
                st.write("Sources:", ", ".join(sources))
                st.json({k: {m: float(v[m]) for m in ["precision","recall","f1-score"]} for k,v in rep.items() if k in LABEL_NAMES})
                st.write("Confusion matrix:", cm)

    # ===== Tester =====
    else:
        st.subheader("🧪 Batch Classifier — Binary Output (CONFIRMED vs FALSE POSITIVE)")
        bundle = load_model_bundle()
        if bundle is None:
            st.warning("No trained model found in `/models`. Go to **Researcher** → Train, then return here.")
        else:
            st.success("Binary model loaded. Ready to classify.")

        st.markdown("Upload a CSV with columns like **period, duration, depth, prad, impact, snr, steff, slogg, srad**. "
                    "KOI/K2/TOI names also work — we auto-map.")

        ctu1, ctu2 = st.columns([1,1])
        with ctu1:
            uploaded = st.file_uploader("Upload CSV to classify", type=["csv"], key="tester_upload")
        with ctu2:
            use_sample = st.button("Use tiny sample (5 rows)", key="tester_use_sample")

        df_in = None
        if uploaded is not None:
            try:
                raw = uploaded.read()
                df_in = safe_read_csv_bytes(raw)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

        if use_sample and df_in is None:
            df_in = pd.DataFrame([{
                "period": 10.5, "duration": 3.4, "depth": 800, "prad": 1.8, "impact": 0.2,
                "snr": 12.0, "steff": 5400, "slogg": 4.5, "srad": 1.0, "mission": "kepler"
            } for _ in range(5)])

        if df_in is None:
            st.info("Upload a CSV (or click the sample button) to begin.")
        else:
            with st.expander("Preview input", expanded=False):
                st.dataframe(df_in.head(20), use_container_width=True, key="tester_preview")
            try:
                results = batch_classify_df(df_in)
                st.success(f"Classified {len(results):,} rows.")
                show = results.copy()
                if "confidence" in show.columns:
                    show["confidence"] = (show["confidence"] * 100).round(2)
                cols = ["period","prad","snr","duration","depth","impact","steff","slogg","srad","mission","prediction","confidence"]
                cols = [c for c in cols if c in show.columns]
                st.dataframe(show[cols], use_container_width=True, key="tester_results_table")
                st.download_button(
                    "⬇️ Download predictions (CSV)",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="predictions_binary.csv", mime="text/csv", key="tester_download_csv"
                )
            except Exception as e:
                st.error("Could not classify this CSV."); st.exception(e)

# ----- Crash guard -----
try:
    main()
except Exception:
    st.error("The app crashed while rendering:")
    st.code("".join(traceback.format_exc()))
