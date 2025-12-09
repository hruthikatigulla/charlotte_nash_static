# =========================
# model_nashville.py
# (Nashville BJJ / CrossFit site selection,
#  mirrored logic from model_approach2.py but using gyms)
# =========================

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import networkx as nx

# ----------- CONSTANTS -----------
CRS_LL = 4326      # WGS84 lat/lon
CRS_M  = 3857      # Web Mercator as a generic metric CRS for Nashville
MILE_M = 1609.344

# ----------- FILES (Nashville data) -----------
BG_FILE       = "nashville_data/nashville_bg_with_income.geojson"
ROADS_FILE    = "nashville_data/nashville_roads_drive.geojson"
BOUNDARY_FILE = "nashville_data/nashville_boundary.geojson"
GYMS_FILE     = "nashville_data/gyms_nashville_clean.geojson"  # all gyms with names


# ------------ utility helpers ------------
def _exists(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)


def _read_to_ll(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf.set_crs(CRS_LL, inplace=True, allow_override=True)
    return gdf.to_crs(CRS_LL)


def _first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def huff_share_vs_competitors(t_new: np.ndarray, t_comps: np.ndarray, beta: float) -> np.ndarray:
    """
    Compute Huff-style share vs competitors, based on travel times.

    t_new:   [Nbg]   travel time from blocks to new candidate (minutes)
    t_comps: [Nbg,K] travel times from blocks to K competitor gyms (minutes)
    beta:    distance-decay exponent
    """
    t_new = np.asarray(t_new, float)
    t_comps = np.asarray(t_comps, float)

    eps = 1e-6  # small positive floor

    # Avoid 0 / negative times safely (no division by zero warnings)
    t_new_safe = np.where(t_new <= 0, eps, t_new)
    t_comps_safe = np.where(t_comps <= 0, np.inf, t_comps)

    A_new = 1.0 / (t_new_safe ** beta)
    A_comp = np.sum(1.0 / (t_comps_safe ** beta), axis=1)

    return A_new / (A_new + A_comp + 1e-9)


# ------------------- ROAD GRAPH HELPERS -------------------
def _build_graph_from_roads(roads_m: gpd.GeoDataFrame):
    if roads_m is None or roads_m.empty:
        return None, None

    G = nx.Graph()
    node_coords = {}

    for geom in roads_m.geometry:
        if geom is None:
            continue

        if geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            lines = [geom]

        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue

            x1, y1 = coords[0]
            x2, y2 = coords[-1]

            dx = x2 - x1
            dy = y2 - y1
            dist = float(np.hypot(dx, dy))
            if not np.isfinite(dist) or dist <= 0:
                continue

            u = (x1, y1)
            v = (x2, y2)

            if G.has_edge(u, v):
                if dist < G[u][v]["weight"]:
                    G[u][v]["weight"] = dist
            else:
                G.add_edge(u, v, weight=dist)

            if u not in node_coords:
                node_coords[u] = (x1, y1)
            if v not in node_coords:
                node_coords[v] = (x2, y2)

    if G.number_of_nodes() == 0:
        return None, None

    print(f"[roads] Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G, node_coords


def _build_road_kdtree(node_coords: dict):
    if not node_coords:
        return None, None, None
    node_ids = list(node_coords.keys())
    coords_array = np.array([node_coords[n] for n in node_ids], dtype=float)
    tree = cKDTree(coords_array)
    return tree, node_ids, coords_array


def _nearest_node(tree, node_ids, x, y):
    dist, idx = tree.query([x, y], k=1)
    return node_ids[int(idx)]


# ------------------- METRIC HELPERS (Nashville) -------------------
def _stores_per_10k(buf_m, stores_m, bg_m) -> float:
    """
    Local competition: gyms per 10k people within a buffer.
    """
    pop_col = _first_existing(
        bg_m.columns,
        ["population", "pop_total", "pop", "POP", "B01001_001E"]
    )
    if pop_col is None:
        return 0.0

    st = stores_m[stores_m.intersects(buf_m)]

    inter = gpd.overlay(
        bg_m[["geometry", pop_col]],
        gpd.GeoDataFrame(geometry=[buf_m], crs=bg_m.crs),
        how="intersection"
    )
    if inter.empty:
        return 0.0

    base = bg_m[["geometry"]].copy()
    base["orig_area"] = base.geometry.area
    inter["part_area"] = inter.geometry.area
    inter = inter.join(base["orig_area"], how="left")
    inter["area_prop"] = (inter["part_area"] / inter["orig_area"].replace(0, np.nan)).clip(0, 1)

    pop_buf = float((inter[pop_col] * inter["area_prop"]).sum())
    if pop_buf <= 0:
        return 0.0

    return len(st) / (pop_buf / 10000.0)


def _accessibility_roadlen(buf_m, roads_m) -> float:
    if roads_m is None or roads_m.empty:
        return 0.0
    cut = roads_m[roads_m.intersects(buf_m)]
    if cut.empty:
        return 0.0
    clipped = gpd.overlay(
        cut,
        gpd.GeoDataFrame(geometry=[buf_m], crs=roads_m.crs),
        how="intersection"
    )
    return float(clipped.length.sum())


def _median_income(buf_m, bg_m) -> float:
    """
    Population-weighted median income within buffer.
    """
    pop_col = _first_existing(
        bg_m.columns,
        ["population", "pop_total", "pop", "POP", "B01001_001E"]
    )
    if pop_col is None:
        return 0.0

    income_col = _first_existing(
        bg_m.columns,
        ["median_income", "income", "med_income", "B19013_001E"]
    )
    if income_col is None:
        return 0.0

    cols = ["geometry", pop_col, income_col]

    inter = gpd.overlay(
        bg_m[cols],
        gpd.GeoDataFrame(geometry=[buf_m], crs=bg_m.crs),
        how="intersection"
    )
    if inter.empty:
        return 0.0

    pop = inter[pop_col].astype(float).values
    vals = inter[income_col].astype(float).values

    mask = np.isfinite(vals) & np.isfinite(pop) & (pop > 0) & (vals > 0)
    if not mask.any():
        return 0.0

    m = float(np.average(vals[mask], weights=pop[mask]))
    return float(np.clip(m, 0, 300000))


# ------------------- DIVERSE SELECTION (same as Charlotte) -------------------
def select_top_diverse(out_m: gpd.GeoDataFrame,
                       scores_col: str = "pair_score",
                       N: int = 10,
                       min_sep_m: float = 3.0 * MILE_M) -> gpd.GeoDataFrame:
    if out_m.empty:
        return out_m

    cand = out_m.sort_values(scores_col, ascending=False).reset_index(drop=True)
    kept_idx = []
    xs = cand.geometry.x.values
    ys = cand.geometry.y.values

    for i in range(len(cand)):
        if len(kept_idx) >= N:
            break
        if not kept_idx:
            kept_idx.append(i)
            continue
        dx = xs[i] - xs[kept_idx]
        dy = ys[i] - ys[kept_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.all(dist >= min_sep_m):
            kept_idx.append(i)

    if not kept_idx:
        kept_idx = [0]
    return cand.iloc[kept_idx].reset_index(drop=True)


# ------------------- LOAD & PREP (Nashville) -------------------
def load_nashville():
    if not _exists(BG_FILE):
        raise FileNotFoundError(f"Missing Nashville BG file: {BG_FILE}")
    if not _exists(GYMS_FILE):
        raise FileNotFoundError(f"Missing gyms file: {GYMS_FILE}")
    if not _exists(ROADS_FILE):
        raise FileNotFoundError(f"Missing Nashville roads file: {ROADS_FILE}")

    # ---- BLOCK GROUPS ----
    bg_ll = _read_to_ll(BG_FILE)

    # population
    pop_col = _first_existing(
        bg_ll.columns,
        ["population", "pop_total", "pop", "POP", "B01001_001E"]
    )
    if pop_col is None:
        raise ValueError(
            "Nashville BG file must have a population-like column "
            "('pop_total', 'population', 'pop', etc.)."
        )
    if pop_col != "population":
        bg_ll = bg_ll.rename(columns={pop_col: "population"})

    # NEW: coerce population to numeric (turn '-', '' etc. into NaN)
    bg_ll["population"] = pd.to_numeric(bg_ll["population"], errors="coerce")

    # income
    inc_col = _first_existing(
        bg_ll.columns,
        ["income", "median_income", "med_income", "B19013_001E"]
    )
    if inc_col is None:
        bg_ll["income"] = 0.0
        inc_col = "income"
    elif inc_col != "income":
        bg_ll = bg_ll.rename(columns={inc_col: "income"})

    # NEW: coerce income to numeric (turn '-', '' etc. into NaN)
    bg_ll["income"] = pd.to_numeric(bg_ll["income"], errors="coerce")

    # ---- ROADS ----
    roads_ll = _read_to_ll(ROADS_FILE)

    # ---- GYMS ----
    gyms_ll = _read_to_ll(GYMS_FILE)
    name_series = gyms_ll.get("name", gyms_ll.get("Name", "")).astype(str).str.lower()

    is_bjj = (
        name_series.str.contains("bjj")
        | name_series.str.contains("jiu jitsu")
        | name_series.str.contains("jiu-jitsu")
        | name_series.str.contains("gracie")
    )
    is_cf = (
        name_series.str.contains("crossfit")
        | name_series.str.contains("cross fit")
    )

    gyms_ll["is_bjj"] = is_bjj
    gyms_ll["is_cf"] = is_cf

    print(
        f"[nashville] gyms total={len(gyms_ll)}, "
        f"bjj={int(is_bjj.sum())}, cf={int(is_cf.sum())}, "
        f"others={int((~(is_bjj | is_cf)).sum())}"
    )

    # ---- BOUNDARY CLIP ----
    try:
        if _exists(BOUNDARY_FILE):
            bound_ll = _read_to_ll(BOUNDARY_FILE)[["geometry"]]
            bound_m = bound_ll.to_crs(CRS_M)
            bound_buf = bound_m.buffer(200).unary_union

            # clip BG
            try:
                bg_ll = gpd.overlay(bg_ll, bound_ll, how="intersection")
            except Exception as e:
                print("[nashville] BG overlay fallback:", e)
                bg_ll = bg_ll[bg_ll.to_crs(CRS_M).intersects(bound_buf)].to_crs(CRS_LL)

            # clip gyms
            gyms_ll = gyms_ll[gyms_ll.to_crs(CRS_M).within(bound_buf)].to_crs(CRS_LL)

            # clip roads
            try:
                roads_ll = gpd.overlay(roads_ll, bound_ll, how="intersection")
            except Exception as e:
                print("[nashville] roads overlay fallback:", e)
                roads_ll = roads_ll[roads_ll.to_crs(CRS_M).intersects(bound_buf)].to_crs(CRS_LL)
    except Exception as e:
        print("[nashville] boundary clip skipped:", e)

    # ---- Project to metric CRS ----
    bg_m = bg_ll.to_crs(CRS_M)
    roads_m = roads_ll.to_crs(CRS_M)[["geometry"]]

    # compute area + density for tooltips
    bg_m["area_sqmi"] = bg_m.geometry.area / (MILE_M ** 2)
    bg_m["pop_per_sqmi"] = bg_m["population"] / bg_m["area_sqmi"].replace(0, np.nan)
    bg_m["median_income"] = bg_m["income"]   # for consistent tooltip key

    gyms_m = gyms_ll.to_crs(CRS_M)
    gyms_m["is_bjj"] = gyms_ll["is_bjj"].values
    gyms_m["is_cf"] = gyms_ll["is_cf"].values

    # candidate sites = BG centroids
    bg_m_cent = bg_m.copy()
    if "cent" not in bg_m_cent.columns:
        bg_m_cent["cent"] = bg_m_cent.geometry.centroid

    cands_m = bg_m_cent[["cent"]].copy()
    cands_m = cands_m.rename(columns={"cent": "geometry"})
    cands_m = gpd.GeoDataFrame(geometry=cands_m["geometry"], crs=CRS_M)
    cands_ll = cands_m.to_crs(CRS_LL)

    # ---- Road graph + KDTree + BG-node mapping ----
    road_graph, node_coords = _build_graph_from_roads(roads_m)
    road_kdtree, road_node_ids, _ = _build_road_kdtree(node_coords)

    bg_xy = np.c_[bg_m_cent["cent"].x.values, bg_m_cent["cent"].y.values]
    dists, idxs = road_kdtree.query(bg_xy, k=1)
    bg_node_ids = [road_node_ids[int(j)] for j in idxs]
    print(f"[nashville] mapped {len(bg_node_ids)} block groups to nearest road nodes.")

    print(
        f"[nashville] BG={len(bg_m)}, cands={len(cands_ll)}, roads={len(roads_m)}"
    )

    # subsets
    bjj_m = gyms_m[gyms_m["is_bjj"]].copy()
    cf_m = gyms_m[gyms_m["is_cf"]].copy()
    others_m = gyms_m[~(gyms_m["is_bjj"] | gyms_m["is_cf"])].copy()

    for gdf in (bjj_m, cf_m, others_m):
        gdf.set_crs(CRS_M, inplace=True, allow_override=True)

    return {
        "bg_m": bg_m,
        "cands_ll": cands_ll,
        "cands_m": cands_m,
        "roads_m": roads_m,
        "road_graph": road_graph,
        "road_kdtree": road_kdtree,
        "road_node_ids": road_node_ids,
        "bg_node_ids": bg_node_ids,
        "gyms_m": gyms_m,
        "bjj_m": bjj_m,
        "cf_m": cf_m,
        "others_m": others_m,
    }


# -------------- core scoring (identical logic to Charlotte) --------------
def _core_score(state_core,
                radius_miles=5.0,
                beta=2.5,
                penalty_lambda=0.25,
                K=3,
                heat_sample=500,
                max_candidates=None,
                topN=10,
                min_sep_miles=3.0,
                W1=1.0, W2=1.0, W3=1.0):

    print(f"[score_all_candidates_like_ht] beta={beta}, lambda={penalty_lambda}, K={K}, "
          f"W1={W1}, W2={W2}, W3={W3}")

    ht_m = state_core["ht_m"]
    comp_m = state_core["comp_m"]
    cands_ll = state_core["cands_ll"].copy()
    cands_m = state_core["cands_m"].copy()
    bg_m = state_core["bg_m"]

    road_graph = state_core.get("road_graph", None)
    road_kdtree = state_core.get("road_kdtree", None)
    road_node_ids = state_core.get("road_node_ids", None)
    bg_node_ids = state_core.get("bg_node_ids", None)

    rad_m = radius_miles * MILE_M

    # cap candidates if needed
    if max_candidates is not None and len(cands_ll) > max_candidates:
        cands_ll = cands_ll.iloc[:max_candidates].reset_index(drop=True)
        cands_m = cands_m.iloc[:max_candidates].reset_index(drop=True)

    cands_ll = cands_ll.reset_index(drop=True)
    cands_m = cands_m.reset_index(drop=True)

    def _norm_weight(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, float).copy()
        a[~np.isfinite(a)] = np.nan
        valid = a > 0
        if not np.any(valid):
            return np.ones_like(a)

        vals = a[valid]
        lo, hi = np.nanpercentile(vals, [1, 99])
        if hi <= lo:
            w = np.ones_like(a)
        else:
            clipped = np.clip(a, lo, hi)
            w = (clipped - lo) / (hi - lo)
        w[~np.isfinite(w)] = 0.0
        return 0.01 + 0.99 * np.clip(w, 0.0, 1.0)

    # demand side
    bg_cent = bg_m.copy()
    if "cent" not in bg_cent.columns:
        bg_cent["cent"] = bg_cent.geometry.centroid

    bg_xy = np.c_[bg_cent["cent"].x.values, bg_cent["cent"].y.values]

    pop_col = _first_existing(
        bg_cent.columns,
        ["population", "pop_total", "pop", "POP", "B01001_001E"]
    )
    bg_pop = bg_cent[pop_col].astype(float).values

    inc_col = _first_existing(
        bg_cent.columns,
        ["income", "median_income", "med_income", "B19013_001E"]
    )
    if inc_col is not None:
        inc = bg_cent[inc_col].astype(float).values
    else:
        inc = np.ones_like(bg_pop, dtype=float)

    pop_w = _norm_weight(bg_pop)
    inc_w = _norm_weight(inc)

    dens_col = _first_existing(
        bg_cent.columns,
        ["pop_per_sqmi", "density", "DENSITY"]
    )
    if dens_col is not None:
        dens = bg_cent[dens_col].astype(float).values
        dens_w = _norm_weight(dens)
    else:
        dens_w = np.ones_like(bg_pop)

    block_weight = 0.4 * pop_w + 0.3 * inc_w + 0.3 * dens_w

    # competitor times
    SPEED_MPS = 35 * MILE_M / 3600.0
    if not comp_m.empty:
        comp_xy = np.c_[comp_m.geometry.x.values, comp_m.geometry.y.values]
        tree = cKDTree(comp_xy)
        k_eff = min(max(1, K), len(comp_xy))
        dists_bg_to_comp, _ = tree.query(bg_xy, k=k_eff)
        if dists_bg_to_comp.ndim == 1:
            dists_bg_to_comp = dists_bg_to_comp[:, None]
        bg_tcomp = (dists_bg_to_comp / SPEED_MPS) / 60.0
    else:
        bg_tcomp = np.full((len(bg_xy), 1), 60.0)

    stores_all_m = gpd.GeoDataFrame(
        geometry=pd.concat([ht_m.geometry, comp_m.geometry], ignore_index=True),
        crs=CRS_M,
    )

    metr_s10k, metr_acc, metr_inc, metr_access_score = [], [], [], []
    metr_potential = []

    cand_xy = np.c_[cands_m.geometry.x.values, cands_m.geometry.y.values]

    for i in range(len(cands_m)):
        cx, cy = cand_xy[i]
        buf = Point(cx, cy).buffer(rad_m)

        s10k = _stores_per_10k(buf, stores_all_m, bg_m)
        acc = _accessibility_roadlen(buf, state_core["roads_m"])
        incM = _median_income(buf, bg_m)

        metr_s10k.append(float(s10k))
        metr_acc.append(float(acc))
        metr_inc.append(float(incM))

        d_new = np.hypot(bg_xy[:, 0] - cx, bg_xy[:, 1] - cy)
        t_new = (d_new / SPEED_MPS) / 60.0
        share = huff_share_vs_competitors(t_new, bg_tcomp, beta)

        potential = float(np.nansum(block_weight * share))
        metr_potential.append(potential)

        access_score = 0.0
        if (road_graph is not None) and (road_kdtree is not None) and \
           (road_node_ids is not None) and (bg_node_ids is not None):

            cand_node = _nearest_node(road_kdtree, road_node_ids, cx, cy)
            lengths = nx.single_source_dijkstra_path_length(
                road_graph, cand_node, weight="weight"
            )

            dist_m = np.full(len(bg_node_ids), np.inf, dtype=float)
            for j, node in enumerate(bg_node_ids):
                if node in lengths:
                    dist_m[j] = float(lengths[node])

            t_road = (dist_m / SPEED_MPS) / 60.0

            valid = np.isfinite(t_road) & (t_road > 0) & \
                    np.isfinite(block_weight) & (block_weight > 0)
            if valid.any():
                w = block_weight[valid]
                weighted_time_min = float(
                    np.nansum(w * t_road[valid]) / np.nansum(w)
                )
                t_hours = max(weighted_time_min, 0.0) / 60.0
                raw = 1.0 / (1.0 + t_hours)
                access_score = float(np.clip(raw, 0.01, 1.0))

        metr_access_score.append(access_score)

    s10k_arr_raw = np.asarray(metr_s10k, float)
    acc_arr = np.asarray(metr_acc, float)
    inc_arr = np.asarray(metr_inc, float)
    access_dj_arr = np.asarray(metr_access_score, float)
    potential_arr_raw = np.asarray(metr_potential, float)

    potential_norm = _norm_weight(potential_arr_raw)

    final_scores = (W1 * potential_norm) \
                   + (W2 * access_dj_arr) \
                   - (W3 * penalty_lambda * s10k_arr_raw)

    ps = np.asarray(final_scores, float)
    inten = (ps - np.nanmin(ps)) / (np.nanmax(ps) - np.nanmin(ps) + 1e-9)

    ps_rounded = np.round(ps, 2)
    s10k_arr_rounded = np.round(s10k_arr_raw, 2)
    acc_arr_rounded = np.round(acc_arr, 2)
    inc_arr_rounded = np.round(inc_arr, 2)
    access_dj_rounded = np.round(access_dj_arr, 2)

    out_ll = cands_ll.copy()
    out_ll["pair_score"] = ps_rounded
    out_ll["intensity"] = inten
    out_ll["stores_per_10k"] = s10k_arr_rounded
    out_ll["access_len_m"] = acc_arr_rounded
    out_ll["income_med"] = inc_arr_rounded
    out_ll["access_score_dj"] = access_dj_rounded

    # score existing brand facilities (BJJ or CF)
    ht_gdf = None
    if (ht_m is not None) and (len(ht_m) > 0):
        ht_scores = []
        ht_s10k_list = []
        ht_acc_list = []
        ht_inc_list = []
        ht_access_list = []

        ht_xy = np.c_[ht_m.geometry.x.values, ht_m.geometry.y.values]

        for hx, hy in ht_xy:
            buf_ht = Point(hx, hy).buffer(rad_m)

            s10k_ht = _stores_per_10k(buf_ht, stores_all_m, bg_m)
            acc_ht = _accessibility_roadlen(buf_ht, state_core["roads_m"])
            incM_ht = _median_income(buf_ht, bg_m)

            d_new_ht = np.hypot(bg_xy[:, 0] - hx, bg_xy[:, 1] - hy)
            t_new_ht = (d_new_ht / SPEED_MPS) / 60.0
            share_ht = huff_share_vs_competitors(t_new_ht, bg_tcomp, beta)
            potential_ht = float(np.nansum(block_weight * share_ht))

            access_score_ht = 0.0
            if (road_graph is not None) and (road_kdtree is not None) and \
               (road_node_ids is not None) and (bg_node_ids is not None):

                ht_node = _nearest_node(road_kdtree, road_node_ids, hx, hy)
                lengths_ht = nx.single_source_dijkstra_path_length(
                    road_graph, ht_node, weight="weight"
                )

                dist_m_ht = np.full(len(bg_node_ids), np.inf, dtype=float)
                for j, node in enumerate(bg_node_ids):
                    if node in lengths_ht:
                        dist_m_ht[j] = float(lengths_ht[node])

                t_road_ht = (dist_m_ht / SPEED_MPS) / 60.0

                valid_ht = np.isfinite(t_road_ht) & (t_road_ht > 0) & \
                           np.isfinite(block_weight) & (block_weight > 0)
                if valid_ht.any():
                    w_ht = block_weight[valid_ht]
                    weighted_time_min_ht = float(
                        np.nansum(w_ht * t_road_ht[valid_ht]) / np.nansum(w_ht)
                    )
                    t_hours_ht = max(weighted_time_min_ht, 0.0) / 60.0
                    raw_ht = 1.0 / (1.0 + t_hours_ht)
                    access_score_ht = float(np.clip(raw_ht, 0.01, 1.0))

            pot_all = np.concatenate([potential_arr_raw, [potential_ht]])
            pot_norm_all = _norm_weight(pot_all)
            P_ht = pot_norm_all[-1]

            S_ht = float(s10k_ht)
            score_ht = (W1 * P_ht) \
                       + (W2 * access_score_ht) \
                       - (W3 * penalty_lambda * S_ht)

            ht_scores.append(score_ht)
            ht_s10k_list.append(s10k_ht)
            ht_acc_list.append(acc_ht)
            ht_inc_list.append(incM_ht)
            ht_access_list.append(access_score_ht)

        if ht_scores:
            ht_scores_arr = np.round(np.asarray(ht_scores, float), 2)
            ht_s10k_arr = np.round(np.asarray(ht_s10k_list, float), 2)
            ht_acc_arr = np.round(np.asarray(ht_acc_list, float), 2)
            ht_inc_arr = np.round(np.asarray(ht_inc_list, float), 2)
            ht_access_arr = np.round(np.asarray(ht_access_list, float), 2)

            print(
                f"[ht_scores] Existing brand facilities: "
                f"n={len(ht_scores_arr)}, "
                f"min={ht_scores_arr.min()}, "
                f"max={ht_scores_arr.max()}, "
                f"mean={np.round(ht_scores_arr.mean(), 2)}"
            )

            ht_gdf = ht_m.copy()
            ht_gdf["pair_score"] = ht_scores_arr
            ht_gdf["stores_per_10k"] = ht_s10k_arr
            ht_gdf["access_len_m"] = ht_acc_arr
            ht_gdf["income_med"] = ht_inc_arr
            ht_gdf["access_score_dj"] = ht_access_arr

    # heat
    heat_df = out_ll.dropna(subset=["intensity"]).copy()
    if len(heat_df) > heat_sample:
        heat_df = heat_df.sort_values("intensity", ascending=False).head(heat_sample)
    heat_points = [
        [float(p.y), float(p.x), float(i)]
        for p, i in zip(heat_df.geometry, heat_df["intensity"])
    ]

    out_m = out_ll.to_crs(CRS_M)
    diverse_top = select_top_diverse(
        out_m[[
            "geometry", "pair_score", "stores_per_10k",
            "access_len_m", "income_med", "access_score_dj"
        ]].copy(),
        scores_col="pair_score",
        N=topN,
        min_sep_m=float(min_sep_miles) * MILE_M,
    )
    top10 = diverse_top.to_crs(CRS_LL).reset_index(drop=True)

    return top10, heat_points, ht_gdf


# -------------- public scoring wrappers --------------
def score_nashville_bjj(state, **kwargs):
    """
    New BJJ gym: BJJ = brand, (CF + others) = competitors.
    """
    comp_m = gpd.GeoDataFrame(
        geometry=pd.concat(
            [state["cf_m"].geometry, state["others_m"].geometry],
            ignore_index=True,
        ),
        crs=CRS_M,
    )
    core_state = {
        "ht_m": state["bjj_m"],
        "comp_m": comp_m,
        "bg_m": state["bg_m"],
        "cands_ll": state["cands_ll"],
        "cands_m": state["cands_m"],
        "roads_m": state["roads_m"],
        "road_graph": state["road_graph"],
        "road_kdtree": state["road_kdtree"],
        "road_node_ids": state["road_node_ids"],
        "bg_node_ids": state["bg_node_ids"],
    }
    return _core_score(core_state, **kwargs)


def score_nashville_cf(state, **kwargs):
    """
    New CrossFit gym: CF = brand, (BJJ + others) = competitors.
    BJJ gyms are included as competitors here.
    """
    comp_m = gpd.GeoDataFrame(
        geometry=pd.concat(
            [state["bjj_m"].geometry, state["others_m"].geometry],
            ignore_index=True,
        ),
        crs=CRS_M,
    )
    core_state = {
        "ht_m": state["cf_m"],
        "comp_m": comp_m,
        "bg_m": state["bg_m"],
        "cands_ll": state["cands_ll"],
        "cands_m": state["cands_m"],
        "roads_m": state["roads_m"],
        "road_graph": state["road_graph"],
        "road_kdtree": state["road_kdtree"],
        "road_node_ids": state["road_node_ids"],
        "bg_node_ids": state["bg_node_ids"],
    }
    return _core_score(core_state, **kwargs)
