"""
geometry.py - 정육면체 격자 / 아파트 직육면체의 면(face) 목록을 만드는 순수 numpy 모듈
(화면 라이브러리와 무관해서 plotly 든 matplotlib 든 그대로 쓸 수 있음)
"""
from functools import lru_cache
import numpy as np
import uhi_core as C

VIEWS = ["weather", "shell", "slice", "full"]
VIEW_LABEL = {"weather": "격자 일기도 (겉면 + 층)", "shell": "껍질 (돔 겉면)",
              "slice": "단면 (슬라이스)", "full": "전체 채움 (무거움)"}

D6 = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
POS = np.array([True, False, True, False, True, False])
CORN = np.array([
    [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
    [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)],
    [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
    [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
    [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
    [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]], dtype=float)

# 면 종류(kind)
K_AIR, K_GROUND, K_TREE, K_AIR_INT, K_BLD = 0, 1, 2, 3, 4


class World:
    def __init__(self):
        W = C.build_world()
        self.mat, self.inside, self.cx, self.blds = W["mat"], W["inside"], W["cx"], W["buildings"]
        self.Z = W["Z"]
        self.ins_p = np.pad(self.inside, 1)
        self.mat_p = np.pad(self.mat, 1)
        self.air_in = (self.inside & (self.mat == C.AIR)).astype(float)
        I, J, K = np.indices(self.mat.shape)
        # 바람 화살표를 그릴 격자 (가로 2칸 간격, 높이 1,3,5,7,9층)
        self.arrow_sel = np.argwhere((self.air_in > 0) & (I % 2 == 0) & (J % 2 == 0) & np.isin(K, [1, 3, 5, 7, 9]))


@lru_cache(maxsize=1)
def get_world():
    return World()


def _dmask(w, cells, kind, cut):
    nb = cells[:, None, :] + D6[None, :, :]
    nbp = nb + 1
    n_in = w.ins_p[nbp[..., 0], nbp[..., 1], nbp[..., 2]]
    n_mat = w.mat_p[nbp[..., 0], nbp[..., 1], nbp[..., 2]]
    n_air = n_in & (n_mat == C.AIR)
    if kind == "air_shell":
        return ~n_in
    if kind == "air_int":
        d = np.zeros_like(n_air)
        d[:, 4] = n_air[:, 4]
        return d
    if kind == "air_full":
        return ~n_in | (n_air & POS[None, :])
    if kind == "air_slice":
        n_slab = nb[..., 1] == cut
        return ~n_in | (n_air & n_slab & POS[None, :]) | (n_air & ~n_slab)
    dm = ~n_in | n_air                                       # 고체 격자
    dm[(cells[:, 2] == 0)[:, None] & (np.arange(6)[None, :] == 5)] = False
    return dm


def _faces(w, cells, dm):
    v = cells[:, None, None, :] + CORN[None, :, :, :]
    verts = v[dm].astype(float)
    verts[..., 0] -= w.cx
    verts[..., 1] -= w.cx
    fc, _ = np.nonzero(dm)
    return verts, cells[fc]


@lru_cache(maxsize=32)
def build_geometry(view, layers, cut):
    """layers = (air, ground, trees, bld) 불리언 튜플.
    반환: [{"kind": int, "verts": (F,4,3), "cells": (F,3) 또는 None, "bid":..., "part":...}, ...]"""
    w = get_world()
    mat, ins = w.mat, w.inside
    air = ins & (mat == C.AIR)
    ground = ins & ((mat == C.ASPHALT) | (mat == C.SOIL))
    trees = ins & ((mat == C.LEAF) | (mat == C.WOOD))
    slab = np.arange(C.NY) != cut
    parts = []
    show_air, show_ground, show_trees, show_bld = layers

    if show_air:
        m = air.copy()
        if view == "slice":
            m[:, slab, :] = False
        cells = np.argwhere(m)
        if view == "weather":
            v, c = _faces(w, cells, _dmask(w, cells, "air_shell", cut))
            parts.append(dict(kind=K_AIR, verts=v, cells=c))
            v, c = _faces(w, cells, _dmask(w, cells, "air_int", cut))
            parts.append(dict(kind=K_AIR_INT, verts=v, cells=c))
        else:
            key = {"shell": "air_shell", "slice": "air_slice", "full": "air_full"}[view]
            v, c = _faces(w, cells, _dmask(w, cells, key, cut))
            parts.append(dict(kind=K_AIR, verts=v, cells=c))
    if show_ground:
        cells = np.argwhere(ground)
        v, c = _faces(w, cells, _dmask(w, cells, "solid", cut))
        parts.append(dict(kind=K_GROUND, verts=v, cells=c))
    if show_trees:
        m = trees.copy()
        if view == "slice":
            m[:, slab, :] = False
        cells = np.argwhere(m)
        if len(cells):
            v, c = _faces(w, cells, _dmask(w, cells, "solid", cut))
            parts.append(dict(kind=K_TREE, verts=v, cells=c))
    if show_bld and len(w.blds):
        bv, bb, bp = [], [], []
        s = C.BLD_SIZE
        for b, (bx, by, h) in enumerate(w.blds):
            x0, x1 = bx - w.cx, bx + s - w.cx
            y0, y1 = by - w.cx, by + s - w.cx
            z0, z1 = 1.0, h + 1.0
            faces = [([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], 0),
                     ([(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)], 0),
                     ([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], 0),
                     ([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], 0),
                     ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], 1)]   # 마지막 = 옥상
            for poly, part in faces:
                bv.append(poly); bb.append(b); bp.append(part)
        parts.append(dict(kind=K_BLD, verts=np.array(bv, dtype=float), cells=None,
                          bid=np.array(bb, dtype=int), part=np.array(bp, dtype=int)))
    return parts


def unique_edges(verts):
    """(F,4,3) 면 목록에서 중복 없는 모서리 목록 (E,2,3) 반환. 좌표는 정수 격자라고 가정."""
    if len(verts) == 0:
        return np.zeros((0, 2, 3))
    iv = np.rint(verts).astype(np.int64)
    a = iv[:, [0, 1, 2, 3], :].reshape(-1, 3)
    b = iv[:, [1, 2, 3, 0], :].reshape(-1, 3)

    def enc(p):
        return (p[:, 0] + 32) * 100000 + (p[:, 1] + 32) * 100 + p[:, 2]

    ea, eb = enc(a), enc(b)
    lo, hi = np.minimum(ea, eb), np.maximum(ea, eb)
    u = np.unique(np.stack([lo, hi], axis=1), axis=0)

    def dec(c):
        x = c // 100000 - 32
        rem = c % 100000
        return np.stack([x, rem // 100 - 32, rem % 100], axis=1)

    return np.stack([dec(u[:, 0]), dec(u[:, 1])], axis=1).astype(float)


def wind_arrows(T, cut=None):
    """온도장 T 에서 화살표 위치와 벡터를 계산. cut 이 주어지면 그 단면(y 인덱스)의 화살표만."""
    w = get_world()
    u, v, ww = C.wind_field(T, w.air_in, w.Z)
    sel = w.arrow_sel
    if cut is not None:
        sel = sel[sel[:, 1] == int(cut)]
    if not len(sel):
        return np.zeros((0, 3)), np.zeros((0, 3))
    idx = tuple(sel.T)
    vec = np.stack([u[idx], v[idx], ww[idx]], axis=1)
    keep = np.linalg.norm(vec, axis=1) > 1e-4
    sel, vec = sel[keep], vec[keep]
    pos = np.stack([sel[:, 0] + 0.5 - w.cx, sel[:, 1] + 0.5 - w.cx, sel[:, 2] + 0.5], axis=1)
    return pos, vec
