"""
도시 열섬(Urban Heat Island) 3D 돔 시뮬레이터  -  Streamlit 웹앱
실행:  streamlit run app.py
"""
import os
import time
import numpy as np
import streamlit as st

import uhi_core as C
import geometry as G
import viz

st.set_page_config(page_title="도시 열섬 3D 돔 시뮬레이터", page_icon="🌡️", layout="wide")


# ---------------------------------------------------------------- 데이터
@st.cache_resource(show_spinner=False)
def get_results():
    """미리 계산해 둔 data/results.npz 를 불러옴. 파일이 없으면 직접 계산(몇 분 걸림)."""
    if os.path.exists(C.DATA_PATH):
        return C.load_results()
    res = C.compute_all(C.DAYS, parallel=True, verbose=False)
    try:
        C.save_results(res)
        res = C.load_results()
    except OSError:
        pass
    return res


@st.cache_resource(show_spinner=False)
def get_range(_results):
    return viz.color_range(_results)


def show_plot(slot, fig):
    """Streamlit 버전에 상관없이 그래프를 컨테이너 너비로 표시."""
    try:
        slot.plotly_chart(fig, width="stretch")
    except TypeError:
        slot.plotly_chart(fig, use_container_width=True)


if not os.path.exists(C.DATA_PATH):
    st.info("처음 실행이라 시뮬레이션을 계산하는 중이야 (1~3분). 다음부터는 바로 열려.")
with st.spinner("시뮬레이션 결과 불러오는 중..."):
    RESULTS = get_results()
CMIN, CMAX = get_range(RESULTS)
N_SNAP = len(RESULTS["concrete"]["hours"])

# ---------------------------------------------------------------- 사이드바 메뉴
sb = st.sidebar
sb.header("메뉴")
mode = sb.radio("옥상 재질", C.ROOF_KEYS, format_func=lambda k: viz.ROOF_NAME[k], index=0)
view = sb.radio("보기 모드", G.VIEWS, format_func=lambda k: G.VIEW_LABEL[k], index=0)
sb.markdown("**표시 레이어**")
l_air = sb.checkbox("공기 격자", True)
l_ground = sb.checkbox("바닥 격자", True)
l_trees = sb.checkbox("나무 격자", True)
l_bld = sb.checkbox("아파트", True)
l_wind = sb.checkbox("바람 화살표", True)
alpha = sb.slider("공기 불투명도", 0.03, 0.8, 0.38, 0.01)
cut = 14
if view == "slice":
    cut = sb.slider("단면 위치 (y)", 1, C.NY - C.PAD - 1, C.NY // 2, 1)
hour = sb.slider("시각 [h]", 0.0, 24.0, 16.0, 0.5)
play = sb.button("▶ 24시간 재생")
sb.caption("마우스로 드래그하면 회전, 휠로 확대/축소, 우클릭 드래그로 이동할 수 있어.")

layers = (l_air, l_ground, l_trees, l_bld)


def render(h):
    """시각 h 에서의 3D 그림과 요약 수치."""
    idx = int(round(h * 2))
    idx = min(idx, N_SNAP - 1)
    r = RESULTS[mode]
    hh = r["hours"][idx]
    title = f"{int(hh):02d}:{int(round((hh % 1) * 60)):02d}  |  {viz.ROOF_NAME[mode]}"
    fig = viz.make_figure(r, idx, view, layers, alpha, cut, l_wind, CMIN, CMAX, title)
    return fig, idx


# ---------------------------------------------------------------- 본문
st.title("🌡️ 도시 열섬 3D 돔 시뮬레이터")
st.caption("반구 돔을 1 m 정육면체 격자로 채우고, 왼쪽은 숲 · 오른쪽은 아스팔트와 아파트로 나눠서 "
           "하루 동안의 온도 변화를 계산한 모형이야. 아파트는 격자가 아니라 직육면체 한 덩어리로 계산돼.")

tab_sim, tab_cmp, tab_info = st.tabs(["3D 시뮬레이션", "옥상 개선 비교", "모형 설명 · 한계"])

with tab_sim:
    r = RESULTS[mode]
    idx0 = min(int(round(hour * 2)), N_SNAP - 1)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("시각", f"{int(r['hours'][idx0]):02d}:{int(round((r['hours'][idx0] % 1) * 60)):02d}")
    m2.metric("열섬 강도 ΔT (도시 − 숲)", f"{r['u_air'][idx0] - r['f_air'][idx0]:.1f} °C")
    m3.metric("옥상 표면 평균", f"{r['rT'][idx0].mean():.1f} °C")
    m4.metric("도시 공기 평균", f"{r['u_air'][idx0]:.1f} °C")

    col_main, col_side = st.columns([1.7, 1])
    with col_main:
        slot3d = st.empty()
    with col_side:
        slot_a, slot_r, slot_d = st.empty(), st.empty(), st.empty()

    def draw(h):
        fig, _ = render(h)
        show_plot(slot3d, fig)
        show_plot(slot_a, viz.chart_air(RESULTS[mode], h))
        show_plot(slot_r, viz.chart_roof(RESULTS, mode, h))
        show_plot(slot_d, viz.chart_dt(RESULTS[mode], h))

    if play:
        for h in np.arange(0.0, 24.01, 0.5):
            draw(float(h))
            time.sleep(0.15)
    else:
        draw(hour)

with tab_cmp:
    rows, base_peak = viz.compare_summary(RESULTS)
    st.subheader("옥상 재질을 바꾸면 얼마나 시원해질까?")
    cols = st.columns(3)
    cols[0].metric("일반 옥상 표면 최고온도", f"{base_peak:.1f} °C")
    for row, col in zip(rows, cols[1:]):
        col.metric(row["name"], f"옥상 최고 {row['roof_peak']:.1f} °C",
                   delta=f"-{row['roof_reduction']:.1f} °C (최대 저감)", delta_color="inverse")
        col.caption(f"도시 공기는 최대 {row['air_reduction']:.2f} °C 낮아짐")
    g1, g2 = st.columns(2)
    show_plot(g1, viz.chart_compare_roof(RESULTS))
    show_plot(g2, viz.chart_compare_lines(RESULTS))
    show_plot(st, viz.chart_compare_bars(rows))
    st.info("옥상 표면은 크게 내려가지만 공기 온도는 조금만 내려가는 이유: 돔 전체에서 옥상이 차지하는 "
            "면적이 작기 때문이야. 옥상 면적 비율을 늘리면 효과도 커져.")

with tab_info:
    st.markdown("""
### 어떤 물리를 계산했나?
격자(복셀) 하나하나에 **에너지 보존(열수지)** 식을 적용했어.

`열용량 × dT/dt = 태양 흡수 − 증발산 − 복사냉각 + 전도/대류 + 인공열`

| 항 | 의미 | 재질에 따라 달라지는 값 |
|---|---|---|
| 태양 흡수 | (1 − **알베도**) × 일사량 | 아스팔트 0.10, 콘크리트 0.30, 쿨루프 0.70 |
| 증발산 냉각 | 식물/녹화 옥상이 흡수 에너지의 일부를 수증기 만드는 데 사용 | 잔디 0.65, 잎 0.80 |
| 열용량 | 재질이 열을 저장하는 능력 | 콘크리트·아스팔트·흙 |
| 전도/대류 | 이웃 격자와 열 교환 | 열전도도, 대류계수 |
| 복사냉각 | 밤에 하늘로 열 방출 | 방사율 |
| 인공열 | 에어컨 실외기 등 | 아파트 벽면에서 배출 |

- **아파트**는 격자가 아니라 직육면체 하나(벽 + 옥상 2개 노드)로 계산하고, 맞닿은 공기 격자와만 열을 주고받아.
  그래서 아파트의 온도 변화는 **주변 격자의 온도 변화**로 나타나.
- **바람**은 온도 차이로 만든 열섬 순환(아래층은 따뜻한 도시로 모이고 위층은 퍼짐)에 숲 → 도시 배경 바람을 더했어.
  이 바람장은 그림용이 아니라 열 이동 계산에도 실제로 쓰여.
- 돔 바깥 공기와는 환기 항으로 열을 교환해.

### 한계 (솔직하게)
- 지표 열활성층 두께(`SOLID_ACTIVE`), 공기 유효 열전도도(`K_AIR`), 환기 시간(`TAU_BG`) 등은 **현실적인 열섬 크기(약 2~6 °C)가
  나오도록 보정한 값**이야. 그래서 절대 온도를 예측하는 모델이 아니라, **재질 차이가 열섬을 어떻게 만드는지 정성적으로 보여 주는 모형**이야.
- 격자 크기가 1 m 인 축소 모형 스케일이고, 일사는 정오 최대 900 W/m² 로 단순화했어.
- 옥상 비교 결과의 신뢰도를 높이려면 **모형 실험(상자 냉각 곡선) 데이터로 `CP`, `ALB` 값을 보정**하고
  기상청 관측 자료와 비교하는 검증 단계가 필요해.

### 값을 바꿔서 다시 계산하려면
`uhi_core.py` 위쪽 설정값을 고치고 `python precompute.py` 를 실행하면 `data/results.npz` 가 새로 만들어져.
""")
