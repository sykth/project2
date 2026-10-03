"""
viz.py - plotly 로 3D 돔(반투명 정육면체 격자 + 바람 화살표)과 2D 그래프를 그리는 함수들
"""
import numpy as np
import plotly.graph_objects as go
import uhi_core as C
import geometry as G

COLORSCALE = "Jet"
BG = "#14141c"
ROOF_NAME = {"concrete": "일반 옥상(콘크리트)", "cool": "쿨루프(알베도 0.7)", "green": "옥상 녹화"}
ROOF_COLOR = {"concrete": "#ff6347", "cool": "#00bfff", "green": "#32cd32"}


def color_range(results):
    """색 범위는 공기 온도에 맞춤 (지표/옥상은 더 뜨거워서 빨강으로 포화)."""
    w = G.get_world()
    mask = w.inside & (w.mat == C.AIR)
    vals = np.concatenate([r["snaps"][:, mask].astype(np.float32).ravel() for r in results.values()])
    return float(np.percentile(vals, 0.5)), float(np.percentile(vals, 99.9))


def _mesh(verts, vals, opacity, name, cmin, cmax, showscale=False):
    F = len(verts)
    base = np.arange(F) * 4
    i = np.concatenate([base, base])
    j = np.concatenate([base + 1, base + 2])
    k = np.concatenate([base + 2, base + 3])
    intensity = np.concatenate([vals, vals])                 # 면(사각형) 하나 = 삼각형 2개
    return go.Mesh3d(
        x=verts[..., 0].ravel(), y=verts[..., 1].ravel(), z=verts[..., 2].ravel(),
        i=i, j=j, k=k, intensity=intensity, intensitymode="cell",
        colorscale=COLORSCALE, cmin=cmin, cmax=cmax, opacity=opacity, name=name,
        flatshading=True, hoverinfo="skip", showscale=showscale,
        lighting=dict(ambient=0.72, diffuse=0.75, specular=0.05, roughness=1.0, fresnel=0.0),
        lightposition=dict(x=60, y=-80, z=120),
        colorbar=dict(title=dict(text="공기 온도 [°C]", font=dict(color="white")),
                      tickfont=dict(color="white"), len=0.6, thickness=14, x=1.0) if showscale else None,
    )


def _lines(verts, color, width, name):
    e = G.unique_edges(verts)
    if len(e) == 0:
        return None
    n = len(e)
    out = [np.full((n, 3), np.nan) for _ in range(3)]
    for ax in range(3):
        out[ax][:, 0] = e[:, 0, ax]
        out[ax][:, 1] = e[:, 1, ax]
    return go.Scatter3d(x=out[0].ravel(), y=out[1].ravel(), z=out[2].ravel(), mode="lines",
                        line=dict(color=color, width=width), hoverinfo="skip", name=name, showlegend=False)


def make_figure(res, idx, view, layers, alpha, cut, show_wind, cmin, cmax, title=""):
    """res: 한 시나리오 결과 dict,  idx: 스냅샷 번호(0~48),  layers=(air, ground, trees, bld)"""
    w = G.get_world()
    parts = G.build_geometry(view, tuple(bool(x) for x in layers), int(cut))
    T = res["snaps"][idx].astype(np.float32)
    traces = []
    first = True
    for p in parts:
        kind, verts = p["kind"], p["verts"]
        if len(verts) == 0:
            continue
        if kind == G.K_BLD:
            vals = np.where(p["part"] == 1, res["rT"][idx][p["bid"]], res["bT"][idx][p["bid"]])
            traces.append(_mesh(verts, vals, 0.82, "아파트", cmin, cmax, showscale=first))
            ln = _lines(verts, "rgba(0,0,0,0.6)", 2, "apt-edge")
        else:
            c = p["cells"]
            vals = T[c[:, 0], c[:, 1], c[:, 2]]
            if kind == G.K_AIR:
                op = min(1.0, alpha * 3) if view == "slice" else alpha
                name = "공기"
            elif kind == G.K_AIR_INT:
                op, name = alpha * 0.55, "공기(층)"
            elif kind == G.K_GROUND:
                op, name = 0.97, "바닥"
            else:
                op, name = 0.92, "나무"
            traces.append(_mesh(verts, vals, op, name, cmin, cmax, showscale=first))
            ln = None
            if kind == G.K_AIR and len(verts) < 9000:
                ln = _lines(verts, "rgba(255,255,255,0.35)", 1, "air-grid")
            elif kind in (G.K_GROUND, G.K_TREE):
                ln = _lines(verts, "rgba(0,0,0,0.35)", 1, "solid-grid")
        first = False
        if ln is not None:
            traces.append(ln)

    if show_wind:
        pos, vec = G.wind_arrows(T, cut if view == "slice" else None)
        if len(pos):
            vec = vec * np.array([1.0, 1.0, 1.5])
            traces.append(go.Cone(
                x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], u=vec[:, 0], v=vec[:, 1], w=vec[:, 2],
                sizemode="scaled", sizeref=0.9, anchor="cm", showscale=False,
                colorscale=[[0, "white"], [1, "white"]], hoverinfo="skip", name="바람"))

    R = C.R
    traces.append(go.Scatter3d(x=[0, 0], y=[-R + 0.5, R - 0.5], z=[1.03, 1.03], mode="lines",
                               line=dict(color="white", width=3, dash="dash"), hoverinfo="skip", showlegend=False))
    traces.append(go.Scatter3d(x=[-R * 0.55, R * 0.55], y=[0, 0], z=[R + 1.8, R + 1.8], mode="text",
                               text=["숲", "아파트 단지"], textfont=dict(size=20, color=["#32cd32", "#ffa500"]),
                               hoverinfo="skip", showlegend=False))
    fig = go.Figure(traces)
    ax = dict(visible=False)
    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=18), x=0.5),
        scene=dict(xaxis=dict(ax, range=[-R, R]), yaxis=dict(ax, range=[-R, R]), zaxis=dict(ax, range=[0, R + 2]),
                   aspectmode="manual", aspectratio=dict(x=1.7, y=1.7, z=0.95), bgcolor=BG,
                   camera=dict(eye=dict(x=1.15, y=-1.45, z=0.85))),
        paper_bgcolor=BG, margin=dict(l=0, r=0, t=40, b=0), height=700, showlegend=False,
        uirevision="uhi-camera")
    return fig


# ---------------------------------------------------------------- 2D 그래프
def _style(fig, title, ytitle, height=260):
    fig.update_layout(title=dict(text=title, font=dict(color="white", size=14)), height=height,
                      paper_bgcolor=BG, plot_bgcolor="#1c1c26", font=dict(color="white"),
                      margin=dict(l=50, r=15, t=40, b=35), legend=dict(orientation="h", y=-0.25),
                      yaxis_title=ytitle)
    fig.update_xaxes(gridcolor="#33334a", title_text="시각 [h]")
    fig.update_yaxes(gridcolor="#33334a")
    return fig


def chart_air(res, hour):
    h = res["hours"]
    fig = go.Figure()
    fig.add_scatter(x=h, y=res["f_air"], name="숲", line=dict(color="#32cd32", width=3))
    fig.add_scatter(x=h, y=res["u_air"], name="도시", line=dict(color="#ffa500", width=3))
    fig.add_scatter(x=h, y=res["amb"], name="바깥 기온", line=dict(color="#87ceeb", dash="dot"))
    fig.add_vline(x=hour, line_color="white", line_width=1)
    return _style(fig, "공기 평균 온도 (지상 1~4 m)", "°C")


def chart_roof(results, mode, hour):
    fig = go.Figure()
    for k, r in results.items():
        fig.add_scatter(x=r["hours"], y=r["rT"].mean(axis=1), name=ROOF_NAME[k],
                        line=dict(color=ROOF_COLOR[k], width=4 if k == mode else 1.5),
                        opacity=1.0 if k == mode else 0.55)
    fig.add_vline(x=hour, line_color="white", line_width=1)
    return _style(fig, "옥상 표면 온도 (평균)", "°C")


def chart_dt(res, hour):
    h = res["hours"]
    dT = res["u_air"] - res["f_air"]
    fig = go.Figure()
    fig.add_scatter(x=h, y=dT, fill="tozeroy", name="ΔT", line=dict(color="#ff6347", width=3),
                    fillcolor="rgba(255,99,71,0.45)", showlegend=False)
    fig.add_vline(x=hour, line_color="white", line_width=1)
    return _style(fig, "열섬 강도 ΔT (도시 - 숲)", "°C")


def compare_summary(results):
    """옥상 개선 효과 표 + 막대그래프용 데이터."""
    base = results["concrete"]
    rows = []
    for k in ("cool", "green"):
        r = results[k]
        roof_d = (base["rT"].mean(axis=1) - r["rT"].mean(axis=1)).max()
        air_d = (base["u_all"] - r["u_all"]).max()
        rows.append(dict(key=k, name=ROOF_NAME[k], roof_peak=float(r["rT"].mean(axis=1).max()),
                         roof_reduction=float(roof_d), air_reduction=float(air_d)))
    return rows, float(base["rT"].mean(axis=1).max())


def chart_compare_lines(results):
    fig = go.Figure()
    for k, r in results.items():
        fig.add_scatter(x=r["hours"], y=r["u_all"], name=ROOF_NAME[k], line=dict(color=ROOF_COLOR[k], width=3))
    fig.add_scatter(x=results["concrete"]["hours"], y=results["concrete"]["amb"], name="바깥 기온",
                    line=dict(color="#87ceeb", dash="dot"))
    return _style(fig, "도시 구역 공기 평균 온도 (전체 높이)", "°C", height=340)


def chart_compare_roof(results):
    fig = go.Figure()
    for k, r in results.items():
        fig.add_scatter(x=r["hours"], y=r["rT"].mean(axis=1), name=ROOF_NAME[k], line=dict(color=ROOF_COLOR[k], width=3))
    return _style(fig, "옥상 표면 온도", "°C", height=340)


def chart_compare_bars(rows):
    fig = go.Figure()
    names = [r["name"] for r in rows]
    fig.add_bar(x=names, y=[r["roof_reduction"] for r in rows], name="옥상 표면", marker_color="#ffa500")
    fig.add_bar(x=names, y=[r["air_reduction"] for r in rows], name="도시 공기", marker_color="#4682b4")
    fig.update_layout(barmode="group")
    return _style(fig, "일반 옥상 대비 최대 온도 저감", "°C", height=340)
