#!/usr/bin/env python3
import os
import math
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from golintou import (
    Params,
    resolve_path,
    read_segments,
    snap_segments,
    classify_nodes_and_segments,
)


class GolintouGUI:
    def __init__(self, master):
        self.master = master
        master.title('Golintou (Python)')

        # Left panel for inputs
        left = tk.Frame(master)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        tk.Label(left, text='vizinhanca').grid(row=0, column=0, sticky='w')
        self.v_entry = tk.Entry(left)
        self.v_entry.insert(0, '0.25')
        self.v_entry.grid(row=0, column=1, padx=6)

        tk.Label(left, text='arquivo').grid(row=1, column=0, sticky='w')
        self.a_entry = tk.Entry(left, width=28)
        self.a_entry.insert(0, 'Frattopo.xyz')
        self.a_entry.grid(row=1, column=1, padx=6)
        tk.Button(left, text='Selecionar...', command=self.select_file).grid(row=1, column=2)

        tk.Label(left, text='discretizacao').grid(row=2, column=0, sticky='w')
        self.d_entry = tk.Entry(left)
        self.d_entry.insert(0, '12')
        self.d_entry.grid(row=2, column=1, padx=6)

        btns = tk.Frame(left)
        btns.grid(row=3, column=0, columnspan=3, pady=10, sticky='w')
        tk.Button(btns, text='Executar', command=self.run).pack(side=tk.LEFT)
        tk.Button(btns, text='Salvar Figura…', command=self.save_figure).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text='Exportar CSVs…', command=self.export_csvs).pack(side=tk.LEFT)

        # Right panel for plots (4x2) and per-plot zoom buttons
        right = tk.Frame(master)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        plot_container = tk.Frame(right)
        plot_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(11, 8.5), dpi=100)
        self.axes = [[None, None], [None, None], [None, None], [None, None]]
        idx = 1
        for i in range(4):
            for j in range(2):
                self.axes[i][j] = self.fig.add_subplot(4, 2, idx)
                idx += 1
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_container)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        # Create per-plot zoom buttons that will be positioned next to each axes
        self._plot_container = plot_container
        self._canvas_widget = canvas_widget
        self.zoom_buttons = {}
        self._create_zoom_buttons()
        # Reposition buttons when canvas resizes and after canvas draws
        canvas_widget.bind('<Configure>', lambda e: self.position_zoom_buttons())
        try:
            self.canvas.mpl_connect('draw_event', lambda evt: self.position_zoom_buttons())
        except Exception:
            pass

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[('XYZ/Texto', '*.*')])
        if path:
            self.a_entry.delete(0, tk.END)
            self.a_entry.insert(0, path)

    def parse_params(self) -> Params:
        try:
            v = float(self.v_entry.get().strip())
        except Exception:
            raise ValueError('vizinhanca deve ser número')
        a = self.a_entry.get().strip()
        try:
            d = int(float(self.d_entry.get().strip()))
        except Exception:
            d = 12
        return Params(vizinhanca=v, arquivo=a, discretizacao=d)

    def run(self):
        try:
            params = self.parse_params()
            arquivo = resolve_path(params.arquivo)
            if not os.path.isfile(arquivo):
                raise FileNotFoundError(f'Arquivo não encontrado: {params.arquivo}')

            raw = read_segments(arquivo)
            segs = snap_segments(raw, params.vizinhanca)
            classes = classify_nodes_and_segments(segs, tol=params.tol)
            self.draw_all(segs, classes, params)
        except Exception as e:
            messagebox.showerror('Erro', str(e))

    def draw_all(self, segs: np.ndarray, classes: dict, params: Params):
        # Unpack
        xi, yi, xf, yf = (segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3])
        I_nodes = classes['I_nodes']
        Y_nodes = classes['Y_nodes']
        X_nodes = classes['X_nodes']
        seg_classes = classes['seg_classes']
        INT_B = classes['INT_B']

        # Helper bounds
        xmin = float(np.min([xi.min(), xf.min()])) if segs.size else 0
        xmax = float(np.max([xi.max(), xf.max()])) if segs.size else 1
        ymin = float(np.min([yi.min(), yf.min()])) if segs.size else 0
        ymax = float(np.max([yi.max(), yf.max()])) if segs.size else 1

        # 1,1: nós I,Y,X
        ax = self.axes[0][0]
        ax.clear()
        for i in range(segs.shape[0]):
            ax.plot([xi[i], xf[i]], [yi[i], yf[i]], 'k', lw=1)
        if I_nodes.size:
            ax.plot(I_nodes[:, 0], I_nodes[:, 1], 'o', mfc='g', mec='g', ms=6)
        if Y_nodes.size:
            ax.plot(Y_nodes[:, 0], Y_nodes[:, 1], '^', mfc='r', mec='r', ms=6)
        if X_nodes.size:
            ax.plot(X_nodes[:, 0], X_nodes[:, 1], 's', mfc='b', mec='b', ms=6)
        ax.set_aspect('equal', 'box')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title('nós I,Y e X', pad=6, fontsize=10)
        ax.tick_params(labelsize=8)

        # 1,2: Proporções I,Y,X (triângulo)
        ax = self.axes[0][1]
        ax.clear()
        nI = len(I_nodes)
        nY = len(Y_nodes)
        nX = len(X_nodes)
        denom = max(nI + nY + nX, 1)
        pI, pY, pX = nI/denom, nY/denom, nX/denom
        Tx = math.cos(math.radians(60)) - pY*math.cos(math.radians(60)) + pX/2
        Ty = math.sin(math.radians(60)) - pY*math.sin(math.radians(60)) - pX*(1/math.tan(math.radians(30)))/2
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
        ax.set_title('proporções I,Y e X', pad=6, fontsize=10)

        # 2,1: Segmentos II/IC/CC
        ax = self.axes[1][0]
        ax.clear()
        for i in range(segs.shape[0]):
            color = dict(II='g', IC='r', CC='b')[seg_classes[i]]
            ax.plot([xi[i], xf[i]], [yi[i], yf[i]], color=color, lw=1)
        ax.set_aspect('equal', 'box')
        ax.set_title('segmentos I-I,I-C e C-C', pad=6, fontsize=10)
        ax.tick_params(labelsize=8)

        # 2,2: Proporções II/IC/CC (triângulo)
        ax = self.axes[1][1]
        ax.clear()
        nII = int(np.sum(seg_classes == 'II'))
        nIC = int(np.sum(seg_classes == 'IC'))
        nCC = int(np.sum(seg_classes == 'CC'))
        denom = max(nII + nIC + nCC, 1)
        pII, pIC, pCC = nII/denom, nIC/denom, nCC/denom
        Txp = math.cos(math.radians(60)) - pIC*math.cos(math.radians(60)) + pCC/2
        Typ = math.sin(math.radians(60)) - pIC*math.sin(math.radians(60)) - pCC*(1/math.tan(math.radians(30)))/2
        tri_x = [0, 0.5, 1, 0]
        tri_y = [0, math.sin(math.radians(60)), 0, 0]
        ax.plot(tri_x, tri_y, 'k', lw=3)
        ax.plot(Txp, Typ, '.r', ms=25)
        ax.text(0.5, 0.95, 'I-I', weight='bold', ha='center')
        ax.text(-0.18, 0.0, 'I-C', weight='bold')
        ax.text(1.05, 0.0, 'C-C', weight='bold')
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.0)
        ax.axis('off')
        ax.set_title('proporções I-I,I-C e C-C', pad=6, fontsize=10)

        # Compute permeability tensors i and ii
        # Degree of each segment (how many intersections)
        deg = INT_B.sum(axis=1)
        # i: using segments with deg >= 2 (if any)
        segs_i = segs[deg >= 2]
        kval, kvec = None, None
        if segs_i.shape[0] > 0:
            rk = np.sqrt((segs_i[:, 2] - segs_i[:, 0])**2 + (segs_i[:, 3] - segs_i[:, 1])**2)
            tk = 1e-4 * rk
            thetak = np.degrees(np.arctan2((segs_i[:, 3] - segs_i[:, 1]), (segs_i[:, 2] - segs_i[:, 0])))
            A = (xmax - xmin) * (ymax - ymin) if (xmax > xmin and ymax > ymin) else 1.0
            P11 = (1/A) * np.sum(rk**2 * tk**3 * (np.sin(np.radians(thetak))**2))
            P12 = (1/A) * np.sum(rk**2 * tk**3 * (np.cos(np.radians(thetak)) * np.sin(np.radians(thetak))))
            P22 = (1/A) * np.sum(rk**2 * tk**3 * (np.cos(np.radians(thetak))**2))
            K = (1/12.0) * np.array([[P11 + P22 - P11, P12 - 0],
                                     [P12 - 0,           P11 + P22 - P22]], dtype=float)
            kval, kvec = np.linalg.eig(K)

        # ii: using all segments scaled by f
        rk = np.sqrt((segs[:, 2] - segs[:, 0])**2 + (segs[:, 3] - segs[:, 1])**2)
        tk = 1e-4 * rk
        thetak = np.degrees(np.arctan2((segs[:, 3] - segs[:, 1]), (segs[:, 2] - segs[:, 0])))
        A = (xmax - xmin) * (ymax - ymin) if (xmax > xmin and ymax > ymin) else 1.0
        n_typex = len(X_nodes)
        n_typey = len(Y_nodes)
        n_typei = len(I_nodes)
        f = max(0.0, (2.94*(4*n_typex+2*n_typey))/(4*n_typex+2*n_typey+n_typei) - 2.13) if (4*n_typex+2*n_typey+n_typei) > 0 else 0.0
        P11 = (1/A) * np.sum(rk**2 * tk**3 * (np.sin(np.radians(thetak))**2))
        P12 = (1/A) * np.sum(rk**2 * tk**3 * (np.cos(np.radians(thetak)) * np.sin(np.radians(thetak))))
        P22 = (1/A) * np.sum(rk**2 * tk**3 * (np.cos(np.radians(thetak))**2))
        K2 = (1/12.0) * np.array([[P11 + P22 - P11, P12 - 0],
                                  [P12 - 0,           P11 + P22 - P22]], dtype=float) * f
        k2val, k2vec = np.linalg.eig(K2)

        # 3,1: razão de permeabilidade i
        ax = self.axes[2][0]
        ax.clear()
        if kval is not None:
            kvalnorm = kval / max(min(kval[0], kval[1]), 1e-12)
            v1 = kvec[:, 0] * kvalnorm[0]
            v2 = kvec[:, 1] * kvalnorm[1]
            ax.plot([0, v1[0]], [0, v1[1]], 'k', lw=1)
            ax.plot([0, -v1[0]], [0, -v1[1]], 'k', lw=1)
            ax.plot([0, v2[0]], [0, v2[1]], 'k', lw=1)
            ax.plot([0, -v2[0]], [0, -v2[1]], 'k', lw=1)
            lim = 0.5 * float(np.max(np.abs([v1[0], v1[1], v2[0], v2[1]]))) + 1e-6
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.text(1, 0.5*lim, f"{float(np.max(kvalnorm)):.3g}")
        ax.set_aspect('equal', 'box')
        ax.set_title('razão de permeabilidade i', pad=6, fontsize=10)
        ax.tick_params(labelsize=8)

        # 3,2: razão de permeabilidade ii
        ax = self.axes[2][1]
        ax.clear()
        k2valnorm = k2val / max(min(k2val[0], k2val[1]), 1e-12)
        v1 = k2vec[:, 0] * k2valnorm[0]
        v2 = k2vec[:, 1] * k2valnorm[1]
        ax.plot([0, v1[0]], [0, v1[1]], 'k', lw=1)
        ax.plot([0, -v1[0]], [0, -v1[1]], 'k', lw=1)
        ax.plot([0, v2[0]], [0, v2[1]], 'k', lw=1)
        ax.plot([0, -v2[0]], [0, -v2[1]], 'k', lw=1)
        lim = 0.5 * float(np.max(np.abs([v1[0], v1[1], v2[0], v2[1]]))) + 1e-6
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.text(1, 0.4*lim, f"{float(np.max(k2valnorm)):.3g}")
        ax.text(1, 0.3*lim, f"f={f:.3g}")
        ax.set_aspect('equal', 'box')
        ax.set_title('razão de permeabilidade ii', pad=6, fontsize=10)
        ax.tick_params(labelsize=8)

        # 4,1: C_B (cell-based, no interpolation)
        ax = self.axes[3][0]
        ax.clear()
        lados = max(int(params.discretizacao), 1)
        prolatlon = (ymax - ymin) / (xmax - xmin) if (xmax > xmin) else 1.0
        nrows = max(int(round(lados * prolatlon)), 1)
        ncols = lados
        CB = np.zeros((nrows, ncols), dtype=float)
        if segs.shape[0] > 0:
            lanco = (xmax - xmin) / ncols if ncols > 0 else (xmax - xmin)
            # centers for display
            lonvec = np.zeros(ncols)
            latvec = np.zeros(nrows)
            for mm in range(nrows-1, -1, -1):
                for nn in range(0, ncols):
                    xilim = xmin + nn * lanco
                    xflim = xmin + (nn + 1) * lanco
                    yilim = ymin + mm * lanco
                    yflim = ymin + (mm + 1) * lanco
                    xmedio = (xilim + xflim) / 2.0
                    ymedio = (yilim + yflim) / 2.0
                    latvec[mm] = ymedio
                    lonvec[nn] = xmedio
                    def in_cell(p):
                        return (xilim <= p[0] <= xflim) and (yilim <= p[1] <= yflim)
                    n_i = sum(in_cell(p) for p in I_nodes)
                    n_y = sum(in_cell(p) for p in Y_nodes)
                    n_x = sum(in_cell(p) for p in X_nodes)
                    if (n_i + n_y + n_x) > 0:
                        CB[mm, nn] = (3*n_y + 4*n_x) / (0.5 * (n_i + 3*n_y + 4*n_x))
                    else:
                        CB[mm, nn] = 0.0
        c = ax.imshow(CB, origin='lower', extent=[xmin, xmax, ymin, ymax], aspect='equal')
        # Colocar a barra de cores do lado de fora, sem alterar o tamanho do eixo
        try:
            cb_ax = inset_axes(
                ax,
                width="6%",
                height="100%",
                loc='lower left',
                bbox_to_anchor=(1.06, 0.0, 1, 1),  # mais afastado do eixo e mais largo
                bbox_transform=ax.transAxes,
                borderpad=0,
            )
            self.fig.colorbar(c, cax=cb_ax, orientation='vertical')
            cb_ax.tick_params(labelsize=8)
        except Exception:
            pass
        ax.set_title('C_{B}', pad=6, fontsize=10)
        ax.tick_params(labelsize=8)

        # 4,2: todos
        ax = self.axes[3][1]
        ax.clear()
        for i in range(segs.shape[0]):
            ax.plot([xi[i], xf[i]], [yi[i], yf[i]], 'k', lw=1)
        ax.set_aspect('equal', 'box')
        ax.set_title('todos', pad=6, fontsize=10)
        ax.tick_params(labelsize=8)

        # store last results for CSV export
        self.last_results = dict(
            segs=segs, I_nodes=I_nodes, Y_nodes=Y_nodes, X_nodes=X_nodes,
            seg_classes=seg_classes, INT_B=INT_B,
            proportions_IYX=(pI, pY, pX), proportions_IIICCC=(pII, pIC, pCC),
            K=(kval, kvec), K2=(k2val, k2vec), f=f,
            CB=CB, extent=(xmin, xmax, ymin, ymax),
        )

        # Manually adjust subplot spacing to avoid overlaps and fit inside GUI
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08,
                                 wspace=0.30, hspace=0.45)
        self.canvas.draw_idle()
        # After drawing, position zoom buttons relative to axes
        self.position_zoom_buttons()

    def _create_zoom_buttons(self):
        # Create buttons in the same order as axes are laid out
        labels = {
            (0,0): 'Zoom nós',
            (0,1): 'Zoom I,Y,X',
            (1,0): 'Zoom segs',
            (1,1): 'Zoom II/IC/CC',
            (2,0): 'Zoom perm i',
            (2,1): 'Zoom perm ii',
            (3,0): 'Zoom C_B',
            (3,1): 'Zoom todos',
        }
        for i in range(4):
            for j in range(2):
                rc = (i, j)
                text = labels.get(rc, 'Zoom')
                # Parent buttons to the canvas widget so coordinates align with pixels
                btn = tk.Button(self._canvas_widget, text=text, width=12,
                                command=lambda rc=rc: self.open_zoom(rc))
                # Place later in position_zoom_buttons()
                btn.place(x=0, y=0)
                self.zoom_buttons[rc] = btn

    def position_zoom_buttons(self):
        # Requires axes to be created and canvas size known
        try:
            cw = self._canvas_widget.winfo_width()
            ch = self._canvas_widget.winfo_height()
            margin = 36  # pixels to the left of axes (avoid covering y-axis)
            # Ensure renderer exists
            try:
                renderer = self.canvas.get_renderer()
            except Exception:
                renderer = None
            for i in range(4):
                for j in range(2):
                    ax = self.axes[i][j]
                    if ax is None:
                        continue
                    # Get axes bbox in display pixels
                    bbox = ax.get_window_extent(renderer=renderer)
                    x0 = bbox.x0
                    y0 = bbox.y0
                    h = bbox.height
                    center_y_display = y0 + h/2.0
                    btn = self.zoom_buttons.get((i,j))
                    if not btn:
                        continue
                    btn_w = btn.winfo_reqwidth()
                    btn_h = btn.winfo_reqheight()
                    x = max(0, int(x0) - margin - btn_w)
                    # Convert display coords (origin bottom-left) to Tk coords (origin top-left)
                    y = max(0, int(ch - center_y_display - btn_h/2))
                    btn.place(x=x, y=y)
        except Exception:
            pass

    def open_zoom(self, rc):
        if not hasattr(self, 'last_results'):
            messagebox.showwarning('Aviso', 'Execute primeiro para gerar resultados.')
            return
        R = self.last_results
        segs = R['segs']
        xi, yi, xf, yf = (segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]) if segs.size else (np.array([]),)*4
        I_nodes = R['I_nodes']
        Y_nodes = R['Y_nodes']
        X_nodes = R['X_nodes']
        seg_classes = R['seg_classes']
        INT_B = R['INT_B']
        (pI, pY, pX) = R['proportions_IYX']
        (pII, pIC, pCC) = R['proportions_IIICCC']
        kval, kvec = R['K']
        k2val, k2vec = R['K2']
        f = R['f']
        CB = R['CB']
        xmin, xmax, ymin, ymax = R['extent']

        win = tk.Toplevel(self.master)
        win.title('Zoom')
        fig = Figure(figsize=(9, 7), dpi=100)
        ax = fig.add_subplot(1, 1, 1)

        def finish(title):
            ax.set_title(title)
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            canvas.draw()

        r, c = rc
        if (r, c) == (0, 0):
            # nós I,Y e X
            for i in range(segs.shape[0]):
                ax.plot([xi[i], xf[i]], [yi[i], yf[i]], 'k', lw=1)
            if I_nodes.size:
                ax.plot(I_nodes[:, 0], I_nodes[:, 1], 'o', mfc='g', mec='g', ms=6)
            if Y_nodes.size:
                ax.plot(Y_nodes[:, 0], Y_nodes[:, 1], '^', mfc='r', mec='r', ms=6)
            if X_nodes.size:
                ax.plot(X_nodes[:, 0], X_nodes[:, 1], 's', mfc='b', mec='b', ms=6)
            ax.set_aspect('equal', 'box')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            finish('nós I,Y e X')
            return
        if (r, c) == (0, 1):
            # I,Y,X triângulo
            Tx = math.cos(math.radians(60)) - pY*math.cos(math.radians(60)) + pX/2
            Ty = math.sin(math.radians(60)) - pY*math.sin(math.radians(60)) - pX*(1/math.tan(math.radians(30)))/2
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
            finish('proporções I,Y e X')
            return
        if (r, c) == (1, 0):
            # segmentos II/IC/CC
            for i in range(segs.shape[0]):
                color = dict(II='g', IC='r', CC='b')[seg_classes[i]]
                ax.plot([xi[i], xf[i]], [yi[i], yf[i]], color=color, lw=1)
            ax.set_aspect('equal', 'box')
            finish('segmentos I-I,I-C e C-C')
            return
        if (r, c) == (1, 1):
            # II/IC/CC triângulo
            Txp = math.cos(math.radians(60)) - pIC*math.cos(math.radians(60)) + pCC/2
            Typ = math.sin(math.radians(60)) - pIC*math.sin(math.radians(60)) - pCC*(1/math.tan(math.radians(30)))/2
            tri_x = [0, 0.5, 1, 0]
            tri_y = [0, math.sin(math.radians(60)), 0, 0]
            ax.plot(tri_x, tri_y, 'k', lw=3)
            ax.plot(Txp, Typ, '.r', ms=25)
            ax.text(0.5, 0.95, 'I-I', weight='bold', ha='center')
            ax.text(-0.18, 0.0, 'I-C', weight='bold')
            ax.text(1.05, 0.0, 'C-C', weight='bold')
            ax.set_aspect('equal', 'box')
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.0)
            ax.axis('off')
            finish('proporções I-I,I-C e C-C')
            return
        if (r, c) == (2, 0):
            # permeabilidade i
            if kval is not None:
                kvalnorm = kval / max(min(kval[0], kval[1]), 1e-12)
                v1 = kvec[:, 0] * kvalnorm[0]
                v2 = kvec[:, 1] * kvalnorm[1]
                ax.plot([0, v1[0]], [0, v1[1]], 'k', lw=1)
                ax.plot([0, -v1[0]], [0, -v1[1]], 'k', lw=1)
                ax.plot([0, v2[0]], [0, v2[1]], 'k', lw=1)
                ax.plot([0, -v2[0]], [0, -v2[1]], 'k', lw=1)
                lim = 0.5 * float(np.max(np.abs([v1[0], v1[1], v2[0], v2[1]]))) + 1e-6
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.text(1, 0.5*lim, f"{float(np.max(kvalnorm)):.3g}")
            ax.set_aspect('equal', 'box')
            finish('razão de permeabilidade i')
            return
        if (r, c) == (2, 1):
            # permeabilidade ii
            k2valnorm = k2val / max(min(k2val[0], k2val[1]), 1e-12)
            v1 = k2vec[:, 0] * k2valnorm[0]
            v2 = k2vec[:, 1] * k2valnorm[1]
            ax.plot([0, v1[0]], [0, v1[1]], 'k', lw=1)
            ax.plot([0, -v1[0]], [0, -v1[1]], 'k', lw=1)
            ax.plot([0, v2[0]], [0, v2[1]], 'k', lw=1)
            ax.plot([0, -v2[0]], [0, -v2[1]], 'k', lw=1)
            lim = 0.5 * float(np.max(np.abs([v1[0], v1[1], v2[0], v2[1]]))) + 1e-6
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.text(1, 0.4*lim, f"{float(np.max(k2valnorm)):.3g}")
            ax.text(1, 0.3*lim, f"f={f:.3g}")
            ax.set_aspect('equal', 'box')
            finish('razão de permeabilidade ii')
            return
        if (r, c) == (3, 0):
            # C_B
            cimg = ax.imshow(CB, origin='lower', extent=[xmin, xmax, ymin, ymax], aspect='equal')
            fig.colorbar(cimg, ax=ax, orientation='vertical')
            finish('C_{B}')
            return
        if (r, c) == (3, 1):
            # todos
            for i in range(segs.shape[0]):
                ax.plot([xi[i], xf[i]], [yi[i], yf[i]], 'k', lw=1)
            ax.set_aspect('equal', 'box')
            finish('todos')
            return

    def save_figure(self):
        if not hasattr(self, 'fig'):
            return
        path = filedialog.asksaveasfilename(defaultextension='.png',
                                            filetypes=[('PNG','*.png'), ('PDF','*.pdf'), ('SVG','*.svg')])
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=200, bbox_inches='tight')
        except Exception as e:
            messagebox.showerror('Erro ao salvar', str(e))

    def export_csvs(self):
        if not hasattr(self, 'last_results'):
            messagebox.showwarning('Aviso', 'Execute primeiro para gerar resultados.')
            return
        outdir = filedialog.askdirectory()
        if not outdir:
            return
        R = self.last_results
        try:
            # Nodes
            np.savetxt(os.path.join(outdir, 'nodes_I.csv'), R['I_nodes'], delimiter=',', header='x,y', comments='', fmt='%.10g')
            np.savetxt(os.path.join(outdir, 'nodes_Y.csv'), R['Y_nodes'], delimiter=',', header='x,y', comments='', fmt='%.10g')
            np.savetxt(os.path.join(outdir, 'nodes_X.csv'), R['X_nodes'], delimiter=',', header='x,y', comments='', fmt='%.10g')
            # Segments with class
            segs = R['segs']
            classes = R['seg_classes']
            segs_with_class = np.column_stack([segs, classes])
            # Save as mixed CSV (use object array -> fallback to manual write)
            with open(os.path.join(outdir, 'segments.csv'), 'w') as f:
                f.write('xi,yi,xf,yf,class\n')
                for i in range(segs.shape[0]):
                    f.write(f"{segs[i,0]:.10g},{segs[i,1]:.10g},{segs[i,2]:.10g},{segs[i,3]:.10g},{classes[i]}\n")
            # Proportions
            pI, pY, pX = R['proportions_IYX']
            pII, pIC, pCC = R['proportions_IIICCC']
            with open(os.path.join(outdir, 'proportions.csv'), 'w') as f:
                f.write('pI,pY,pX,pII,pIC,pCC\n')
                f.write(f"{pI:.10g},{pY:.10g},{pX:.10g},{pII:.10g},{pIC:.10g},{pCC:.10g}\n")
            # Permeability i
            kval, kvec = R['K'] if R['K'][0] is not None else (np.array([np.nan, np.nan]), np.eye(2))
            np.savetxt(os.path.join(outdir, 'permeability_i_matrix.csv'), (kvec @ np.diag(kval) @ np.linalg.inv(kvec)), delimiter=',', fmt='%.10g')
            np.savetxt(os.path.join(outdir, 'permeability_i_eigvals.csv'), kval.reshape(1, -1), delimiter=',', header='lambda1,lambda2', comments='', fmt='%.10g')
            np.savetxt(os.path.join(outdir, 'permeability_i_eigvecs.csv'), kvec, delimiter=',', fmt='%.10g')
            # Permeability ii
            k2val, k2vec = R['K2']
            np.savetxt(os.path.join(outdir, 'permeability_ii_matrix.csv'), (k2vec @ np.diag(k2val) @ np.linalg.inv(k2vec)), delimiter=',', fmt='%.10g')
            np.savetxt(os.path.join(outdir, 'permeability_ii_eigvals.csv'), k2val.reshape(1, -1), delimiter=',', header='lambda1,lambda2', comments='', fmt='%.10g')
            np.savetxt(os.path.join(outdir, 'permeability_ii_eigvecs.csv'), k2vec, delimiter=',', fmt='%.10g')
            with open(os.path.join(outdir, 'permeability_ii_info.csv'), 'w') as f:
                f.write('f\n')
                f.write(f"{R['f']:.10g}\n")
            # CB grid and extent
            np.savetxt(os.path.join(outdir, 'CB.csv'), R['CB'], delimiter=',', fmt='%.10g')
            xmin, xmax, ymin, ymax = R['extent']
            with open(os.path.join(outdir, 'CB_extent.csv'), 'w') as f:
                f.write('xmin,xmax,ymin,ymax,nrows,ncols\n')
                nrows, ncols = R['CB'].shape
                f.write(f"{xmin:.10g},{xmax:.10g},{ymin:.10g},{ymax:.10g},{nrows},{ncols}\n")
            # Intersections (if present)
            # Re-compute from INT_B and segs midpoints captured in classes if necessary
            # Here, export adjacency as 0/1 matrix for completeness
            np.savetxt(os.path.join(outdir, 'intersections_adjacency.csv'), R['INT_B'].astype(int), delimiter=',', fmt='%d')
            messagebox.showinfo('Exportação concluída', f'CSVs salvos em:\n{outdir}')
        except Exception as e:
            messagebox.showerror('Erro ao exportar', str(e))


def main():
    root = tk.Tk()
    app = GolintouGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
