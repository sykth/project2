"""
uhi_core.py  -  도시 열섬(UHI) 3D 돔 시뮬레이션 핵심 계산 (웹/화면 코드와 분리된 순수 numpy 모듈)

- build_world : 돔 / 숲(격자) / 아파트(직육면체) 배치
- wind_field  : 온도장 -> 열섬 순환 바람장
- simulate    : 열수지 방정식 적분 (일사, 증발산, 전도/대류, 복사냉각, 인공열, 환기, 바람)
- save_results / load_results : 계산 결과를 data/results.npz 로 저장/불러오기
"""
import os
import numpy as np

# ----------------------------------------------------------------------
# 1. 설정값 (숫자를 바꿔 가며 실험해 보세요)
# ----------------------------------------------------------------------
R = 12                 # 돔 반지름 [격자 수]
PAD = 2
NX = NY = 2 * R + 2 * PAD
NZ = R + 3
DX = 1.0               # 격자 한 변 [m]  (모형 스케일)
DT = 5.0               # 시간 간격 [s]
DAYS = 2               # 첫날은 예열, 둘째 날을 보여줌
SNAP_EVERY_S = 1800    # 저장 간격 30분

T_MEAN, T_AMP = 26.0, 5.0     # 바깥 기온 평균/진폭 [C]
S0 = 900.0                    # 정오 일사량 [W/m^2]
SKY_DELTA = 15.0
SIGMA = 5.670e-8
H_CONV = 10.0                 # 지면/옥상 대류 열전달계수 [W/m^2K]
H_WALL = 14.0                 # 벽면 (대류+복사)
K_AIR = 20.0                  # 공기 유효 열전도도(난류 혼합 포함) [W/mK]
WALL_FACTOR = 0.35            # 벽이 받는 평균 일사 비율
SOLID_ACTIVE = 0.15           # 지면 복셀의 열활성층 보정
WIND_U = 0.015                # 배경 바람: 숲 -> 도시 방향 [m/s] (모형 스케일)
C_G = 0.012                   # 열섬 순환: 수평 온도기울기 -> 바람 [m/s per (K/m)]
U_MAX = 0.03                  # 수평 바람 최대 [m/s]
C_W = 0.004                   # 더운 공기 상승: 온도 편차 -> 상승 바람 [m/s per K]
W_MAX = 0.02                  # 상승 바람 최대 [m/s]
WIND_EVERY = 20               # 바람장 갱신 주기 [step]
TAU_VENT = 100.0              # 돔 껍질 환기 시간 [s]
TAU_BG = 150.0                # 돔 내부 배경 환기 시간 [s]
Q_ANTH_FACE = 25.0            # 아파트 벽면 1칸당 인공열(에어컨 실외기 등) [W]
TREE_SPACING = 3.6
BLD_SIZE = 3                  # 아파트 가로/세로 [칸]
WALL_ALBEDO = 0.30
WALL_ACTIVE = 0.12            # 벽/옥상 열활성 두께 [m]
G_ROOF_BODY = 5.0             # 옥상층-건물 본체 사이 열전달 [W/m^2K]

AIR, ASPHALT, SOIL, LEAF, WOOD, BLD = range(6)
#               air   asph   soil   leaf   wood   bld
CP   = np.array([1200, 1.9e6, 2.0e6, 1.0e6, 1.4e6, 1.0])
KP   = np.array([K_AIR, 0.75, 1.2,   0.5,   0.2,   0.0])
ALB  = np.array([0.0,  0.10, 0.20,  0.20,  0.25,  0.0])
ET   = np.array([0.0,  0.0,  0.65,  0.80,  0.0,   0.0])
EPS  = np.array([0.0,  0.95, 0.95,  0.95,  0.90,  0.0])

# 옥상 재질: (알베도, 증발산 비율, 방사율, 두께[m], 부피열용량[J/m^3K])
ROOFS = {"concrete": (0.30, 0.0, 0.90, 0.05, 2.1e6),
         "cool":     (0.70, 0.0, 0.90, 0.05, 2.1e6),
         "green":    (0.20, 0.65, 0.95, 0.10, 2.0e6)}
ROOF_KEYS = ["concrete", "cool", "green"]
ROOF_LABEL = {"concrete": ("일반 옥상(콘크리트)", "Plain concrete roof"),
              "cool": ("쿨루프(알베도 0.7)", "Cool roof (albedo 0.7)"),
              "green": ("옥상 녹화", "Green roof")}
ROOF_COLOR = {"concrete": "tomato", "cool": "deepskyblue", "green": "limegreen"}




# ----------------------------------------------------------------------
# 2. 월드: 돔 / 숲(격자) / 아파트(직육면체)
# ----------------------------------------------------------------------
def build_world(seed=3):
    rng = np.random.default_rng(seed)
    mat = np.zeros((NX, NY, NZ), dtype=int)
    cx = NX / 2
    xs = np.arange(NX) + 0.5
    zs = np.arange(NZ) + 0.5
    X, Y, Z = np.meshgrid(xs, xs, zs, indexing="ij")
    inside = (X - cx) ** 2 + (Y - cx) ** 2 + Z ** 2 <= R ** 2

    mat[:, :, 0] = SOIL
    mat[X[:, :, 0] > cx, 0] = ASPHALT

    buildings = []
    for bx in range(int(cx) + 1, NX - PAD, BLD_SIZE + 2):
        for by in range(PAD, NY - PAD - BLD_SIZE + 1, BLD_SIZE + 2):
            h = int(rng.integers(6, 10))
            while h >= 4 and not inside[bx:bx + BLD_SIZE, by:by + BLD_SIZE, h].all():
                h -= 1
            if h < 4:
                continue
            mat[bx:bx + BLD_SIZE, by:by + BLD_SIZE, 1:h + 1] = BLD
            buildings.append((bx, by, h))

    trees = []
    tries = 0
    while len(trees) < 16 and tries < 800:
        tries += 1
        tx = int(rng.integers(PAD, int(cx) - 1))
        ty = int(rng.integers(PAD, NY - PAD))
        if any((tx - a) ** 2 + (ty - b) ** 2 < TREE_SPACING ** 2 for a, b in trees):
            continue
        trunk = int(rng.integers(2, 4))
        if not inside[tx, ty, trunk + 2]:
            continue
        trees.append((tx, ty))
        for z in range(1, trunk + 1):
            mat[tx, ty, z] = WOOD
        for z in (trunk + 1, trunk + 2):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    x, y = tx + dx, ty + dy
                    if inside[x, y, z] and mat[x, y, z] == AIR:
                        mat[x, y, z] = LEAF
    return dict(mat=mat, inside=inside, buildings=buildings, cx=cx, X=X, Y=Y, Z=Z)


# ----------------------------------------------------------------------
# 3. 시뮬레이션 (열수지 방정식)
# ----------------------------------------------------------------------
def sl(ax, s):
    idx = [slice(None)] * 3
    idx[ax] = s
    return tuple(idx)


def t_amb(h):
    return T_MEAN + T_AMP * np.sin(2 * np.pi * (h - 9) / 24)


def solar(h):
    return S0 * max(0.0, np.sin(np.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0.0


def anth_profile(h):
    return 0.35 + 0.65 * np.exp(-((h - 18) / 4.0) ** 2)


def _smooth(A):
    for _ in range(2):
        A = (A + np.roll(A, 1, 0) + np.roll(A, -1, 0) + np.roll(A, 1, 1) + np.roll(A, -1, 1)) / 5.0
    return A


def wind_field(T, air_in, Z):
    """온도장에서 바람장을 계산 (열섬 순환).
    - 같은 높이의 평균보다 따뜻한 곳으로 아래층 공기가 모여들고(수렴), 위층에서는 퍼져 나감
    - 따뜻한 공기는 위로, 차가운 공기는 아래로 움직임
    - 여기에 숲 -> 도시 방향의 배경 바람이 더해짐"""
    a = air_in > 0
    cnt = a.sum(axis=(0, 1))
    Tbar = (T * a).sum(axis=(0, 1)) / np.maximum(cnt, 1)
    Tp = _smooth(np.where(a, T - Tbar[None, None, :], 0.0))
    gx = np.gradient(Tp, DX, axis=0)
    gy = np.gradient(Tp, DX, axis=1)
    f = np.cos(np.pi * Z / (R * 0.9))                   # 아래: +1(수렴), 위: 음(발산)
    u = WIND_U + C_G * f * gx
    v = C_G * f * gy
    sp = np.sqrt(u ** 2 + v ** 2)
    k = np.where(sp > U_MAX, U_MAX / np.maximum(sp, 1e-12), 1.0)
    u, v = u * k, v * k
    w = np.clip(C_W * Tp, -W_MAX, W_MAX) * np.clip((Z - 0.5) / 2.5, 0, 1) * np.where(Z > R - 3, 0.3, 1.0)
    return u * a, v * a, w * a


def simulate(roof, days=DAYS, verbose=False):
    W = build_world()
    mat, inside, buildings, cx = W["mat"], W["inside"], W["buildings"], W["cx"]
    X, Y, Z = W["X"], W["Y"], W["Z"]
    air = mat == AIR
    bld = mat == BLD
    solid = (mat > 0) & ~bld
    Ceff = CP[mat] * np.where(solid, SOLID_ACTIVE, 1.0) * DX ** 3

    # (a) 격자-격자 열전도 [W/K]
    LO = [sl(a, slice(None, -1)) for a in range(3)]
    HI = [sl(a, slice(1, None)) for a in range(3)]
    G = []
    for a in range(3):
        k1, k2 = KP[mat][LO[a]], KP[mat][HI[a]]
        kf = np.where(k1 + k2 > 0, 2 * k1 * k2 / (k1 + k2 + 1e-30), 0.0)
        mixed = solid[LO[a]] != solid[HI[a]]
        kf = np.where(mixed, H_CONV * DX, kf)
        kf = np.where(bld[LO[a]] | bld[HI[a]], 0.0, kf)    # 아파트는 격자 전도 없음
        G.append(kf * DX)

    # (b) 지면/나무의 일사, 그늘, 복사
    above = np.zeros_like(mat)
    above[:, :, :-1] = mat[:, :, 1:]
    top_exposed = solid & (above == AIR)
    tau = np.where(air, 1.0, np.where(mat == LEAF, 0.25, np.where(mat == WOOD, 0.5, 0.0)))
    cum = np.cumprod(tau[:, :, ::-1], axis=2)
    trans_above = np.concatenate([np.ones((NX, NY, 1)), cum[:, :, :-1]], axis=2)[:, :, ::-1]
    side_air = sum((np.roll(mat, s, axis=a) == AIR).astype(float) for a in (0, 1) for s in (1, -1))
    A_sun = (1 - ALB[mat]) * DX ** 2 * (top_exposed * trans_above + WALL_FACTOR * side_air) * solid * inside
    A_net = A_sun * (1 - ET[mat])
    A_rad = EPS[mat] * DX ** 2 * top_exposed * trans_above * inside

    # (c) 아파트 <-> 주변 공기 격자 접촉면
    nb = len(buildings)
    bid = -np.ones(mat.shape, dtype=int)
    for b, (bx, by, h) in enumerate(buildings):
        bid[bx:bx + BLD_SIZE, by:by + BLD_SIZE, 1:h + 1] = b
    ax_l, ay_l, az_l, cb_l, cr_l = [], [], [], [], []
    for (dx, dy, dz, is_roof) in [(1, 0, 0, 0), (-1, 0, 0, 0), (0, 1, 0, 0), (0, -1, 0, 0), (0, 0, 1, 1)]:
        nbm = np.roll(mat, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        ix, iy, iz = np.nonzero(bld & (nbm == AIR))
        ax_l.append(ix + dx); ay_l.append(iy + dy); az_l.append(iz + dz)
        cb_l.append(bid[ix, iy, iz]); cr_l.append(np.full(len(ix), bool(is_roof)))
    c_idx = (np.concatenate(ax_l), np.concatenate(ay_l), np.concatenate(az_l))
    c_b = np.concatenate(cb_l)
    c_roof = np.concatenate(cr_l)
    wall_area = np.bincount(c_b[~c_roof], minlength=nb).astype(float) * DX ** 2
    roof_area = np.bincount(c_b[c_roof], minlength=nb).astype(float) * DX ** 2
    alb_r, et_r, eps_r, th_r, cv_r = ROOFS[roof]
    C_body = 2.1e6 * WALL_ACTIVE * (wall_area + roof_area)
    C_roof = cv_r * th_r * roof_area
    G_rb = G_ROOF_BODY * roof_area
    h_c = np.where(c_roof, H_CONV, H_WALL) * DX ** 2

    # (d) 환기/바람
    dist = R - np.sqrt((X - cx) ** 2 + (Y - cx) ** 2 + Z ** 2)
    shell = (inside & air & (dist < 2.5)).astype(float)
    air_in = (inside & air).astype(float)
    AM = [np.roll(air_in, 1, axis=a) > 0 for a in range(3)]
    AP = [np.roll(air_in, -1, axis=a) > 0 for a in range(3)]

    T = np.full(mat.shape, t_amb(0.0))
    Tb = np.full(nb, t_amb(0.0))
    Tr = np.full(nb, t_amb(0.0))
    nsteps = int(days * 86400 / DT)
    snap_every = int(SNAP_EVERY_S / DT)
    t_show = (days - 1) * 86400
    ped = inside & air & (Z < 4.5) & (Z > 1)
    f_mask = ped & (X < cx)
    u_mask = ped & (X > cx)
    uall_mask = inside & air & (X > cx)

    snaps, bT, rT, hours, f_air, u_air, u_all, amb = [], [], [], [], [], [], [], []
    for n in range(nsteps + 1):
        t = n * DT
        h = (t / 3600.0) % 24
        Ta = t_amb(h)
        S = solar(h)
        if n % snap_every == 0 and t >= t_show:
            snaps.append(T.astype(np.float32).copy())
            bT.append(Tb.copy()); rT.append(Tr.copy())
            hours.append(h if h > 0 or len(hours) == 0 else 24.0)
            f_air.append(T[f_mask].mean()); u_air.append(T[u_mask].mean())
            u_all.append(T[uall_mask].mean()); amb.append(Ta)
        if n == nsteps:
            break

        Q = np.zeros_like(T)
        for a in range(3):
            F = G[a] * (T[HI[a]] - T[LO[a]])
            Q[LO[a]] += F
            Q[HI[a]] -= F
        Q += S * A_net
        Q -= A_rad * SIGMA * ((T + 273.15) ** 4 - (Ta - SKY_DELTA + 273.15) ** 4)

        # 아파트 <-> 공기
        Tsurf = np.where(c_roof, Tr[c_b], Tb[c_b])
        fl = h_c * (Tsurf - T[c_idx])                      # +면 공기가 데워짐
        np.add.at(Q, c_idx, fl + np.where(c_roof, 0.0, Q_ANTH_FACE * anth_profile(h)))
        out_w = np.bincount(c_b, weights=np.where(c_roof, 0.0, fl), minlength=nb)
        out_r = np.bincount(c_b, weights=np.where(c_roof, fl, 0.0), minlength=nb)
        cond = G_rb * (Tr - Tb)
        Qb = S * WALL_FACTOR * (1 - WALL_ALBEDO) * wall_area - out_w + cond
        Qr = (S * (1 - alb_r) * (1 - et_r) * roof_area - out_r - cond
              - eps_r * SIGMA * ((Tr + 273.15) ** 4 - (Ta - SKY_DELTA + 273.15) ** 4) * roof_area)

        T += DT * Q / Ceff
        Tb += DT * Qb / C_body
        Tr += DT * Qr / C_roof

        # 바람(이류), 환기, 바깥 경계
        if n % WIND_EVERY == 0:
            uvw = wind_field(T, air_in, Z)
        for a in range(3):                                  # 바람에 의한 열 이동(풍상 차분)
            vel = uvw[a]
            Tm = np.where(AM[a], np.roll(T, 1, axis=a), T)
            Tp_ = np.where(AP[a], np.roll(T, -1, axis=a), T)
            T -= air_in * (DT / DX) * np.where(vel > 0, vel * (T - Tm), vel * (Tp_ - T))
        T += shell * (DT / TAU_VENT) * (Ta - T)
        T += air_in * (DT / TAU_BG) * (Ta - T)
        T[~inside] = Ta

        if verbose and n % (nsteps // 10) == 0:
            print(f"  [{roof}] {100 * n // nsteps:3d}%", flush=True)

    return dict(roof=roof, snaps=np.array(snaps), bT=np.array(bT), rT=np.array(rT),
                hours=np.array(hours), f_air=np.array(f_air), u_air=np.array(u_air),
                u_all=np.array(u_all), amb=np.array(amb))



# ----------------------------------------------------------------------
# 4. 결과 저장 / 불러오기
# ----------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results.npz")
_KEYS = ["snaps", "bT", "rT", "hours", "f_air", "u_air", "u_all", "amb"]


def _run(args):
    return simulate(*args)


def compute_all(days=DAYS, parallel=True, verbose=True):
    """일반/쿨루프/옥상녹화 세 시나리오를 계산 (코어가 여러 개면 병렬)."""
    jobs = [(k, days, False) for k in ROOF_KEYS]
    if parallel:
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=3) as ex:
                return dict(zip(ROOF_KEYS, ex.map(_run, jobs)))
        except Exception as e:                                # 병렬이 막힌 환경이면 순차 실행
            if verbose:
                print("병렬 실행 실패 -> 순차 실행:", e)
    return {k: simulate(k, days, verbose) for k in ROOF_KEYS}


def save_results(results, path=DATA_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {}
    for k, r in results.items():
        for name in _KEYS:
            arr = np.asarray(r[name])
            out[f"{k}__{name}"] = arr.astype(np.float16) if name == "snaps" else arr.astype(np.float32)
    np.savez_compressed(path, **out)


def load_results(path=DATA_PATH):
    z = np.load(path)
    return {k: {name: z[f"{k}__{name}"] for name in _KEYS} | {"roof": k} for k in ROOF_KEYS}
