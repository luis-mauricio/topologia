#!/usr/bin/env python3
"""
Golintou (Python) – Análise Topológica de Lineamentos

This initial Python port focuses on:
- Reading the 4-column XYZ-like file with segments (xi yi xf yf)
- Applying a snapping/rounding factor (vizinhanca)
- Detecting intersections between segments (touch or cross)
- Classifying node types I (isolated ends), Y (end-point connections), X (true crossings)
- Classifying segments as I-I, I-C, C-C (based on whether ends touch an intersection)
- Plotting a triangular I–Y–X diagram and segment views

Notes:
- This is not yet a 1:1 feature parity with the MATLAB UI (Golintou.m).
- It’s a solid base that can be extended to match more outputs if desired.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Params:
    vizinhanca: float = 0.25
    arquivo: str = "Frattopo.xyz"
    discretizacao: int = 12  # kept for compatibility; not used yet
    tol: float = 1e-9        # tolerance for float comparisons


def resolve_path(path: str) -> str:
    # Strip surrounding quotes if any
    p = path.strip()
    if p and ((p[0] == p[-1] == '"') or (p[0] == p[-1] == "'")):
        p = p[1:-1]
    # Try current dir, else alongside this script
    if os.path.isfile(p):
        return p
    here = os.path.dirname(os.path.abspath(__file__))
    alt = os.path.join(here, p)
    return alt


def read_segments(path: str) -> np.ndarray:
    """Reads a text file with 4 float columns: xi yi xf yf. Returns (N,4) array."""
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:
        if data.size != 4:
            raise ValueError(f"Arquivo {path} deve conter 4 colunas")
        data = data.reshape(1, 4)
    if data.shape[1] != 4:
        raise ValueError(f"Arquivo {path} deve conter 4 colunas, encontrado {data.shape[1]}")
    return data


def snap_segments(segments: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 0:
        return segments.copy()
    snapped = np.round(segments / factor) * factor
    return snapped


def _pairwise_intersections(xi, yi, xf, yf, tol=1e-12):
    """Vectorized pairwise intersection computation.

    Returns:
    - INT_B: (N,N) boolean mask for segments that intersect/touch within extents
    - INT_X, INT_Y: (N,N) intersection coordinates
    - PAR_B: (N,N) boolean mask for parallel pairs
    """
    xi = np.asarray(xi)[:, None]
    yi = np.asarray(yi)[:, None]
    xf = np.asarray(xf)[:, None]
    yf = np.asarray(yf)[:, None]

    x3 = np.asarray(xi).T
    y3 = np.asarray(yi).T
    x4 = np.asarray(xf).T
    y4 = np.asarray(yf).T

    X1 = np.repeat(xi, xi.shape[1] if xi.ndim > 1 else x3.shape[1], axis=1)
    Y1 = np.repeat(yi, yi.shape[1] if yi.ndim > 1 else y3.shape[1], axis=1)
    X2 = np.repeat(xf, xf.shape[1] if xf.ndim > 1 else x4.shape[1], axis=1)
    Y2 = np.repeat(yf, yf.shape[1] if yf.ndim > 1 else y4.shape[1], axis=1)

    X3 = np.repeat(x3, x3.shape[0] if x3.ndim > 1 else xi.shape[0], axis=0)
    Y3 = np.repeat(y3, y3.shape[0] if y3.ndim > 1 else yi.shape[0], axis=0)
    X4 = np.repeat(x4, x4.shape[0] if x4.ndim > 1 else xf.shape[0], axis=0)
    Y4 = np.repeat(y4, y4.shape[0] if y4.ndim > 1 else yf.shape[0], axis=0)

    # Differences
    X4_X3 = X4 - X3
    Y4_Y3 = Y4 - Y3
    X2_X1 = X2 - X1
    Y2_Y1 = Y2 - Y1
    X1_X3 = X1 - X3
    Y1_Y3 = Y1 - Y3

    numerator_a = X4_X3 * Y1_Y3 - Y4_Y3 * X1_X3
    numerator_b = X2_X1 * Y1_Y3 - Y2_Y1 * X1_X3
    denominator = Y4_Y3 * X2_X1 - X4_X3 * Y2_Y1

    PAR_B = np.isclose(denominator, 0.0, atol=tol)
    with np.errstate(divide='ignore', invalid='ignore'):
        u_a = np.where(~PAR_B, numerator_a / denominator, np.nan)
        u_b = np.where(~PAR_B, numerator_b / denominator, np.nan)

    INT_X = X1 + X2_X1 * u_a
    INT_Y = Y1 + Y2_Y1 * u_a
    INT_B = (~PAR_B) & (u_a >= -tol) & (u_a <= 1 + tol) & (u_b >= -tol) & (u_b <= 1 + tol)
    np.fill_diagonal(INT_B, False)

    return INT_B, INT_X, INT_Y, PAR_B


def unique_points(points: np.ndarray, tol=1e-9) -> np.ndarray:
    """Return unique points within a tolerance, preserving order."""
    if points.size == 0:
        return points.reshape(0, 2)
    uniq = []
    for p in points:
        if not any(np.allclose(p, q, atol=tol) for q in uniq):
            uniq.append(p)
    return np.array(uniq)


def classify_nodes_and_segments(segments: np.ndarray, tol=1e-9):
    """Classify nodes into I/Y/X and segments into I-I / I-C / C-C.

    Returns dict with keys:
      xp, yp: intersection points arrays (K,)
      I_nodes, Y_nodes, X_nodes: arrays (Mi,2), (My,2), (Mx,2)
      seg_classes: array of shape (N,), values in {"II","IC","CC"}
    """
    xi, yi, xf, yf = (segments[:, 0], segments[:, 1], segments[:, 2], segments[:, 3])

    INT_B, INT_X, INT_Y, PAR_B = _pairwise_intersections(xi, yi, xf, yf, tol=tol)

    # Gather intersection points (n<=m) where INT_B True
    xs = []
    ys = []
    N = segments.shape[0]
    for n in range(N):
        for m in range(n, N):
            if INT_B[n, m]:
                xs.append(INT_X[n, m])
                ys.append(INT_Y[n, m])
    xp = np.array(xs)
    yp = np.array(ys)
    XYint = np.vstack([xp, yp]).T if xp.size else np.zeros((0, 2))
    XYint = unique_points(XYint, tol=tol)

    # Endpoints
    XYstart = segments[:, 0:2]
    XYend = segments[:, 2:4]

    # Y nodes: intersection points that coincide with any segment endpoint
    Y_nodes = []
    for p in XYint:
        if (np.any(np.all(np.isclose(XYstart, p, atol=tol), axis=1)) or
                np.any(np.all(np.isclose(XYend, p, atol=tol), axis=1))):
            Y_nodes.append(p)
    Y_nodes = unique_points(np.array(Y_nodes), tol=tol)

    # X nodes: intersection points that are NOT endpoints
    if XYint.size:
        X_nodes = []
        for p in XYint:
            if not (np.any(np.all(np.isclose(XYstart, p, atol=tol), axis=1)) or
                    np.any(np.all(np.isclose(XYend, p, atol=tol), axis=1))):
                X_nodes.append(p)
        X_nodes = unique_points(np.array(X_nodes), tol=tol)
    else:
        X_nodes = np.zeros((0, 2))

    # I nodes: endpoints not in XYint
    endpoints = np.vstack([XYstart, XYend])
    I_nodes = []
    for p in endpoints:
        if not np.any(np.all(np.isclose(XYint, p, atol=tol), axis=1)):
            I_nodes.append(p)
    I_nodes = unique_points(np.array(I_nodes), tol=tol)

    # Segment classes based on whether start/end coincide with any intersection point
    seg_classes = np.empty(N, dtype=object)
    for i in range(N):
        s = XYstart[i]
        e = XYend[i]
        s_conn = np.any(np.all(np.isclose(XYint, s, atol=tol), axis=1))
        e_conn = np.any(np.all(np.isclose(XYint, e, atol=tol), axis=1))
        if s_conn and e_conn:
            seg_classes[i] = "CC"
        elif s_conn or e_conn:
            seg_classes[i] = "IC"
        else:
            seg_classes[i] = "II"

    return dict(
        xp=xp, yp=yp,
        I_nodes=I_nodes, Y_nodes=Y_nodes, X_nodes=X_nodes,
        seg_classes=seg_classes,
        INT_B=INT_B
    )


def plot_results(segments: np.ndarray, classes: dict, title_prefix: str = ""):
    xi, yi, xf, yf = (segments[:, 0], segments[:, 1], segments[:, 2], segments[:, 3])
    I_nodes = classes["I_nodes"]
    Y_nodes = classes["Y_nodes"]
    X_nodes = classes["X_nodes"]
    seg_classes = classes["seg_classes"]

    # Ternary I-Y-X diagram point
    nI = len(I_nodes)
    nY = len(Y_nodes)
    nX = len(X_nodes)
    denom = max(nI + nY + nX, 1)
    pI, pY, pX = nI / denom, nY / denom, nX / denom
    Tx = math.cos(math.radians(60)) - pY * math.cos(math.radians(60)) + pX / 2.0
    Ty = math.sin(math.radians(60)) - pY * math.sin(math.radians(60)) - pX * (1 / math.tan(math.radians(30))) / 2.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Ax1: I-Y-X triangle
    ax = axes[0, 0]
    tri_x = [0, 0.5, 1, 0]
    tri_y = [0, math.sin(math.radians(60)), 0, 0]
    ax.plot(tri_x, tri_y, 'k', lw=3)
    ax.plot(Tx, Ty, '.r', ms=25)
    ax.text(0.5, 0.95, 'I', weight='bold', ha='center')
    ax.text(-0.1, 0.0, 'Y', weight='bold')
    ax.text(1.05, 0.0, 'X', weight='bold')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.axis('off')
    ax.set_title(f"Proporções I,Y,X – I:{nI} Y:{nY} X:{nX}")

    # Ax2: Segments colored by II/IC/CC
    ax = axes[0, 1]
    for i in range(segments.shape[0]):
        color = dict(II='g', IC='r', CC='b')[seg_classes[i]]
        ax.plot([xi[i], xf[i]], [yi[i], yf[i]], color=color, lw=1)
    ax.set_aspect('equal', 'box')
    ax.set_title('Segmentos I-I (g), I-C (r), C-C (b)')

    # Ax3: All segments
    ax = axes[1, 0]
    for i in range(segments.shape[0]):
        ax.plot([xi[i], xf[i]], [yi[i], yf[i]], 'k', lw=1)
    ax.set_aspect('equal', 'box')
    ax.set_title('Todos os segmentos')

    # Ax4: Nodes I/Y/X scatter
    ax = axes[1, 1]
    if I_nodes.size:
        ax.plot(I_nodes[:, 0], I_nodes[:, 1], 'o', mfc='g', mec='g', ms=6, label='I')
    if Y_nodes.size:
        ax.plot(Y_nodes[:, 0], Y_nodes[:, 1], '^', mfc='r', mec='r', ms=6, label='Y')
    if X_nodes.size:
        ax.plot(X_nodes[:, 0], X_nodes[:, 1], 's', mfc='b', mec='b', ms=6, label='X')
    ax.legend(loc='best')
    ax.set_aspect('equal', 'box')
    ax.set_title('Nós I (g), Y (r), X (b)')

    fig.suptitle(title_prefix + ' Golintou (Python)')
    fig.tight_layout()
    plt.show()


def main():
    p = argparse.ArgumentParser(description="Golintou (Python) – Análise Topológica de Lineamentos")
    p.add_argument('-f', '--arquivo', default='Frattopo.xyz', help='Arquivo com colunas xi yi xf yf')
    p.add_argument('-v', '--vizinhanca', type=float, default=0.25, help='Fator de arredondamento (snap)')
    p.add_argument('-d', '--discretizacao', type=int, default=12, help='Compatibilidade; não usado ainda')
    p.add_argument('--tol', type=float, default=1e-9, help='Tolerância para comparações')
    args = p.parse_args()

    params = Params(vizinhanca=args.vizinhanca, arquivo=args.arquivo,
                    discretizacao=args.discretizacao, tol=args.tol)

    arquivo = resolve_path(params.arquivo)
    if not os.path.isfile(arquivo):
        raise FileNotFoundError(f"Arquivo não encontrado: {params.arquivo}")

    raw = read_segments(arquivo)
    segs = snap_segments(raw, params.vizinhanca)
    classes = classify_nodes_and_segments(segs, tol=params.tol)
    plot_results(segs, classes, title_prefix=os.path.basename(arquivo) + ' – ')


if __name__ == '__main__':
    main()
