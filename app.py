import streamlit as st
import heapq
import time
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# 1. DATA KAMPUS
# ============================================================
graph = {
    "Gerbang Utama":      {"Rektorat": 200, "Fakultas Teknik": 300},
    "Rektorat":           {"Gerbang Utama": 200, "Perpustakaan": 150, "Fakultas Teknik": 250, "Fakultas Ekonomi": 200},
    "Fakultas Teknik":    {"Gerbang Utama": 300, "Rektorat": 250, "Fakultas MIPA": 200, "Kantin Pusat": 180},
    "Perpustakaan":       {"Rektorat": 150, "Fakultas Ekonomi": 120, "Fakultas MIPA": 220},
    "Fakultas Ekonomi":   {"Rektorat": 200, "Perpustakaan": 120, "Fakultas Hukum": 160},
    "Fakultas MIPA":      {"Fakultas Teknik": 200, "Perpustakaan": 220, "Kantin Pusat": 100, "Fakultas Hukum": 190},
    "Fakultas Hukum":     {"Fakultas Ekonomi": 160, "Fakultas MIPA": 190, "Kantin Pusat": 140},
    "Kantin Pusat":       {"Fakultas Teknik": 180, "Fakultas MIPA": 100, "Fakultas Hukum": 140},
}

koordinat = {
    "Gerbang Utama":    (0, 0),
    "Rektorat":         (2, 1),
    "Fakultas Teknik":  (1, 4),
    "Perpustakaan":     (4, 2),
    "Fakultas Ekonomi": (5, 1),
    "Fakultas MIPA":    (3, 5),
    "Fakultas Hukum":   (6, 3),
    "Kantin Pusat":     (4, 6),
}

# ============================================================
# 2. HEURISTIC
# ============================================================
def heuristic(node, goal):
    x1, y1 = koordinat[node]
    x2, y2 = koordinat[goal]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# ============================================================
# 3. ALGORITMA A*
# ============================================================
def a_star(start, goal, graph, heuristic):
    open_set = []
    heapq.heappush(open_set, (0, start))
    g_score = {n: float("inf") for n in graph}
    g_score[start] = 0
    f_score = {n: float("inf") for n in graph}
    f_score[start] = heuristic(start, goal)
    parent = {start: None}
    closed_set = set()
    log = []
    langkah = 0

    while open_set:
        current_f, current = heapq.heappop(open_set)
        if current in closed_set:
            continue

        langkah += 1
        g = g_score[current]
        h = heuristic(current, goal)
        f = g + h

        log.append({
            "langkah": langkah, "node": current,
            "g": round(g, 2), "h": round(h, 2), "f": round(f, 2),
            "open": [(n, round(fs, 2)) for fs, n in open_set if n not in closed_set],
            "closed": sorted(closed_set),
        })

        if current == goal:
            break

        closed_set.add(current)

        for tetangga, biaya in graph[current].items():
            if tetangga in closed_set:
                continue
            g_baru = g_score[current] + biaya
            if g_baru < g_score[tetangga]:
                parent[tetangga] = current
                g_score[tetangga] = g_baru
                f_score[tetangga] = g_baru + heuristic(tetangga, goal)
                heapq.heappush(open_set, (f_score[tetangga], tetangga))

    if goal not in parent:
        return None, None, langkah, log

    jalur = []
    node = goal
    while node is not None:
        jalur.append(node)
        node = parent[node]
    jalur.reverse()
    return jalur, g_score[goal], langkah, log

# ============================================================
# 4. KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Algoritma A*",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️Pencarian Jalur Kampus Dengan Algoritma A*")
st.write("Pilih lokasi awal dan tujuan, lalu klik **Cari Jalur Optimal**.")

# ============================================================
# 5. SIDEBAR: INPUT USER
# ============================================================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    semua_node = sorted(graph.keys())
    start = st.selectbox("📍 Titik Awal", semua_node, index=0)
    goal  = st.selectbox("🎯 Titik Tujuan", semua_node, index=len(semua_node) - 1)
    st.divider()
    tampilkan_log = st.checkbox("Tampilkan proses langkah demi langkah", value=False)
    st.divider()
    st.caption("Universitas Jember")
    st.caption("Fakultas Ilmu Komputer")

# ============================================================
# 6. VALIDASI
# ============================================================
if start == goal:
    st.warning("⚠️ Titik awal dan tujuan tidak boleh sama.")
    st.stop()

# ============================================================
# 7. TOMBOL & EKSEKUSI
# ============================================================
if st.button("🔍 Cari Jalur Optimal", type="primary", use_container_width=True):
    mulai = time.time()
    jalur, biaya, langkah, log = a_star(start, goal, graph, heuristic)
    durasi = time.time() - mulai

    if jalur is None:
        st.error("❌ Tidak ada jalur dari start ke goal.")
    else:
        st.success(f"🎯 Jalur ditemukan dalam {durasi*1000:.2f} ms")

        # --- Metrik ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Biaya", f"{biaya:.0f} m")
        col2.metric("Jumlah Pindah", f"{len(jalur) - 1} kali")
        col3.metric("Node Dieksplor", f"{langkah} node")
        col4.metric("Waktu", f"{durasi*1000:.2f} ms")

        # --- Jalur Teks ---
        st.subheader("🛣️ Jalur Optimal")
        st.write(" → ".join(f"**{n}**" for n in jalur))

        # --- Visualisasi Peta ---
        st.subheader("🗺️ Visualisasi Peta")
        G = nx.Graph()
        for u, neighbors in graph.items():
            for v, w in neighbors.items():
                G.add_edge(u, v, weight=w)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Semua edge (abu-abu)
        nx.draw_networkx_edges(G, koordinat, ax=ax,
                               edge_color="lightgray", width=2)
        # Edge jalur optimal (merah)
        path_edges = list(zip(jalur, jalur[1:]))
        nx.draw_networkx_edges(G, koordinat, edgelist=path_edges, ax=ax,
                               edge_color="red", width=4)
        # Semua node (biru)
        nx.draw_networkx_nodes(G, koordinat, ax=ax,
                               node_color="skyblue", node_size=1500)
        # Start hijau, Goal merah
        nx.draw_networkx_nodes(G, koordinat, nodelist=[start], ax=ax,
                               node_color="lime", node_size=1800)
        nx.draw_networkx_nodes(G, koordinat, nodelist=[goal], ax=ax,
                               node_color="tomato", node_size=1800)
        # Label
        nx.draw_networkx_labels(G, koordinat, ax=ax, font_size=8)

        ax.set_title(f"Jalur Optimal: {start} → {goal}")
        ax.axis("off")
        st.pyplot(fig)

        # --- Log Langkah ---
        if tampilkan_log:
            st.subheader("📜 Proses Langkah demi Langkah")

            df = pd.DataFrame([
                {"Langkah": item["langkah"], "Node": item["node"],
                 "g": item["g"], "h": item["h"], "f": item["f"]}
                for item in log
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.write("**Detail tiap langkah:**")
            for item in log:
                with st.expander(f"Langkah {item['langkah']}: ambil **{item['node']}** (f={item['f']})"):
                    st.write(f"g = {item['g']}  |  h = {item['h']}  |  f = {item['f']}")
                    st.write(f"**Open set:** {item['open']}")
                    st.write(f"**Closed set:** {item['closed']}")