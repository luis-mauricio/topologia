#!/usr/bin/env python3
import os
import math
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.font as tkfont

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
from matplotlib.colors import ListedColormap

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
        # Disable window maximization (user cannot maximize the main window)
        try:
            master.resizable(False, False)
        except Exception:
            pass

        # Left panel for inputs
        left = tk.Frame(master)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        # Allow centering of the buttons row across the three columns
        try:
            left.grid_columnconfigure(0, weight=1)
            left.grid_columnconfigure(1, weight=1)
            left.grid_columnconfigure(2, weight=1)
        except Exception:
            pass
        self.left = left

        tk.Label(left, text='Vizinhança').grid(row=0, column=0, sticky='w')
        self.v_entry = tk.Entry(left, width=28)
        self.v_entry.insert(0, '0.25')
        self.v_entry.grid(row=0, column=1, padx=6)

        tk.Label(left, text='Arquivo').grid(row=1, column=0, sticky='w')
        self.a_entry = tk.Entry(left, width=28)
        self.a_entry.insert(0, 'Frattopo.xyz')
        self.a_entry.grid(row=1, column=1, padx=6)
        self.select_btn = tk.Button(left, text='Selecionar', command=self.select_file)
        self.select_btn.grid(row=1, column=2)

        tk.Label(left, text='Discretização').grid(row=2, column=0, sticky='w')
        self.d_entry = tk.Entry(left, width=28)
        self.d_entry.insert(0, '12')
        self.d_entry.grid(row=2, column=1, padx=6)

        btns = tk.Frame(left)
        # Center the button group within the left panel
        btns.grid(row=3, column=0, columnspan=3, pady=10)
        self.btn_run = tk.Button(btns, text='Executar', command=self.run)
        self.btn_save = tk.Button(btns, text='Salvar Figura', command=self.save_figure)
        self.btn_export = tk.Button(btns, text='Exportar CSVs', command=self.export_csvs)
        # Use a 3-column grid so we can enforce equal pixel widths
        for col in range(3):
            try:
                btns.grid_columnconfigure(col, weight=1, uniform='btns_col')
            except Exception:
                pass
        self.btn_run.grid(row=0, column=0, sticky='ew', padx=6)
        self.btn_save.grid(row=0, column=1, sticky='ew', padx=6)
        self.btn_export.grid(row=0, column=2, sticky='ew', padx=6)
        # After layout, harmonize the three buttons' widths (and match Selecionar)
        try:
            self.master.after(0, self._sync_button_widths)
        except Exception:
            pass

        # Track selected full path (shown as basename only)
        self._selected_fullpath = None

        # Defer plot area creation until the user clicks Executar
        self.right = None
        self.fig = None
        self.axes = None
        self.canvas = None
        self._plot_container = None
        self._canvas_widget = None
        self.zoom_buttons = {}

        # Ensure a 'parula' colormap is available by loading/parsing CSV (generate if missing)
        try:
            self._ensure_parula_colormap()
        except Exception:
            # Fallback: alias to viridis
            try:
                base = cm.get_cmap('viridis', 256)
                parula_cmap = ListedColormap(base(np.linspace(0, 1, 256)), name='parula')
                matplotlib.colormaps.register(parula_cmap, name='parula')
            except Exception:
                pass
        # Start with a compact window showing only the left panel (no extra right space)
        try:
            self.master.update_idletasks()
            lw = self.left.winfo_reqwidth()
            lh = self.left.winfo_reqheight()
            # padding minimal e simétrico com o restante da UI
            pad_w, pad_h = 16, 20
            self.master.geometry(f"{lw+pad_w}x{lh+pad_h}")
        except Exception:
            pass

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[('XYZ/Texto', '*.*')])
        if path:
            self._selected_fullpath = path
            self.a_entry.delete(0, tk.END)
            self.a_entry.insert(0, os.path.basename(path))

        # Also resync widths in case system font metrics changed
        try:
            self.master.after(0, self._sync_button_widths)
        except Exception:
            pass

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
            # Prefer the full path selected via dialog (but display only basename)
            if self._selected_fullpath and os.path.basename(self._selected_fullpath) == os.path.basename(params.arquivo.strip().strip("'\"")):
                arquivo = self._selected_fullpath
            else:
                arquivo = resolve_path(params.arquivo)
            if not os.path.isfile(arquivo):
                raise FileNotFoundError(f'Arquivo não encontrado: {params.arquivo}')

            raw = read_segments(arquivo)
            segs = snap_segments(raw, params.vizinhanca)
            classes = classify_nodes_and_segments(segs, tol=params.tol)
            # Lazily create the plot area and expand the window
            if self.right is None:
                self.init_plot_area()
                # Fit window tightly to requested sizes
                self.set_window_to_requested_size()
            self.draw_all(segs, classes, params)
        except Exception as e:
            messagebox.showerror('Erro', str(e))

    def init_plot_area(self):
        if self.right is not None:
            return
        # Right panel for plots (4x2) and per-plot zoom buttons
        right = tk.Frame(self.master)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.right = right
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
        canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
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

    def draw_all(self, segs: np.ndarray, classes: dict, params: Params):
        def _subdivide_segments_by_intersections(segments: np.ndarray, XYint: np.ndarray, tol: float = 1e-9) -> np.ndarray:
            if segments.size == 0:
                return segments.reshape(0, 4)
            pieces = []
            for i in range(segments.shape[0]):
                x0, y0, x1, y1 = segments[i]
                v = np.array([x1 - x0, y1 - y0], dtype=float)
                vv = float(v[0]*v[0] + v[1]*v[1])
                if vv <= tol*tol:
                    continue
                # Collect t values for intersections that lie on this segment
                tvals = [0.0, 1.0]
                if XYint.size:
                    for p in XYint:
                        px, py = float(p[0]), float(p[1])
                        # Bounding box quick reject (expanded by tol)
                        if not (min(x0, x1) - tol <= px <= max(x0, x1) + tol and
                                min(y0, y1) - tol <= py <= max(y0, y1) + tol):
                            continue
                        # Check colinearity via cross product magnitude
                        cross = v[0]*(py - y0) - v[1]*(px - x0)
                        if abs(cross) > tol * max(1.0, math.sqrt(vv)):
                            continue
                        # Parametric position along the segment
                        t = ((px - x0)*v[0] + (py - y0)*v[1]) / vv
                        if -1e-12 <= t <= 1 + 1e-12:
                            tvals.append(min(1.0, max(0.0, t)))
                # Unique and sorted
                tvals = sorted(set(tvals))
                # Create pieces between successive t values
                for a, b in zip(tvals[:-1], tvals[1:]):
                    if (b - a) <= 1e-12:
                        continue
                    xs = x0 + a * v[0]
                    ys = y0 + a * v[1]
                    xe = x0 + b * v[0]
                    ye = y0 + b * v[1]
                    if (abs(xe - xs) <= tol and abs(ye - ys) <= tol):
                        continue
                    pieces.append([xs, ys, xe, ye])
            return np.array(pieces, dtype=float) if pieces else np.zeros((0, 4), dtype=float)

        # Unpack
        xi, yi, xf, yf = (segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3])
        I_nodes = classes['I_nodes']
        Y_nodes = classes['Y_nodes']
        X_nodes = classes['X_nodes']
        seg_classes = classes['seg_classes']
        INT_B = classes['INT_B']
        # Intersection points as (K,2) for downstream logic
        if 'xp' in classes and 'yp' in classes and classes['xp'].size:
            XYint = np.vstack([classes['xp'], classes['yp']]).T
        else:
            XYint = np.zeros((0, 2), dtype=float)

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
        handles = []
        if I_nodes.size:
            hI, = ax.plot(I_nodes[:, 0], I_nodes[:, 1], 'o', mfc='g', mec='g', ms=6, label='I')
            handles.append(hI)
        if Y_nodes.size:
            hY, = ax.plot(Y_nodes[:, 0], Y_nodes[:, 1], '^', mfc='r', mec='r', ms=6, label='Y')
            handles.append(hY)
        if X_nodes.size:
            hX, = ax.plot(X_nodes[:, 0], X_nodes[:, 1], 's', mfc='b', mec='b', ms=6, label='X')
            handles.append(hX)
        # Match 'todos' plot behavior: let Matplotlib autoscale and add a small margin
        ax.set_aspect('equal', 'box')
        try:
            ax.margins(x=0.05, y=0.05)
        except Exception:
            pass
        ax.set_title('nós I,Y e X', pad=6, fontsize=10)
        ax.tick_params(labelsize=8)
        # Legend outside on the right
        if handles:
            try:
                # Position legend slightly off to the right
                ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.06, 1.0),
                          borderaxespad=0.0, fontsize=8)
            except Exception:
                pass

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
        # Reference guides (geologic = magenta, tectonic = dashed black) as in MATLAB
        try:
            # Geologic (5 segments), normalized by 190
            Mnosgeol = np.array([
                [15.805556, 27.805556, 93.000000, 162.500000],
                [93.000000, 162.500000, 107.250000, 126.027778],
                [107.250000, 126.027778, 111.944444, 85.222222],
                [111.944444, 85.222222, 92.444444, 48.750000],
                [92.444444, 48.750000, 15.805556, 27.805556],
            ]) / 190.0
            for row in Mnosgeol:
                ax.plot([row[0], row[2]], [row[1], row[3]], 'm', lw=2)
        except Exception:
            pass
        try:
            # Tectonic (2 segments), normalized by 308
            Mnostect = np.array([
                [117.348688, 197.476647, 185.069738, 207.298784],
                [95.636596, 161.806782, 204.197057, 175.764555],
            ]) / 308.0
            for row in Mnostect:
                ax.plot([row[0], row[2]], [row[1], row[3]], '--k', lw=2)
        except Exception:
            pass
        ax.plot(Tx, Ty, '.r', ms=12)
        ax.text(0.5, 0.95, 'I', weight='bold', ha='center')
        ax.text(-0.1, 0.0, 'Y', weight='bold')
        ax.text(1.05, 0.0, 'X', weight='bold')
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.0)
        ax.axis('off')
        ax.set_title('proporções I,Y e X', pad=6, fontsize=10)

        # 2,1: Segmentos II/IC/CC (baseado em subsegmentos, como no MATLAB)
        ax = self.axes[1][0]
        ax.clear()
        # Subdivide original segmentos nos pontos de interseção
        segs_sub = _subdivide_segments_by_intersections(segs, XYint, tol=params.tol)
        # Classificar subsegmentos conforme extremidades conectadas a interseções
        seg_classes_sub = np.empty(segs_sub.shape[0], dtype=object)
        for i in range(segs_sub.shape[0]):
            s = segs_sub[i, 0:2]
            e = segs_sub[i, 2:4]
            s_conn = XYint.size and np.any(np.all(np.isclose(XYint, s, atol=params.tol), axis=1))
            e_conn = XYint.size and np.any(np.all(np.isclose(XYint, e, atol=params.tol), axis=1))
            if s_conn and e_conn:
                seg_classes_sub[i] = 'CC'
            elif s_conn or e_conn:
                seg_classes_sub[i] = 'IC'
            else:
                seg_classes_sub[i] = 'II'
        for i in range(segs_sub.shape[0]):
            color = dict(II='g', IC='r', CC='b')[seg_classes_sub[i]]
            ax.plot([segs_sub[i,0], segs_sub[i,2]], [segs_sub[i,1], segs_sub[i,3]], color=color, lw=1)
        ax.set_aspect('equal', 'box')
        ax.set_title('segmentos I-I,I-C e C-C', pad=6, fontsize=10)
        ax.tick_params(labelsize=8)

        # 2,2: Proporções II/IC/CC (triângulo) com base nos subsegmentos
        ax = self.axes[1][1]
        ax.clear()
        nII = int(np.sum(seg_classes_sub == 'II'))
        nIC = int(np.sum(seg_classes_sub == 'IC'))
        nCC = int(np.sum(seg_classes_sub == 'CC'))
        denom = max(nII + nIC + nCC, 1)
        pII, pIC, pCC = nII/denom, nIC/denom, nCC/denom
        Txp = math.cos(math.radians(60)) - pIC*math.cos(math.radians(60)) + pCC/2
        Typ = math.sin(math.radians(60)) - pIC*math.sin(math.radians(60)) - pCC*(1/math.tan(math.radians(30)))/2
        tri_x = [0, 0.5, 1, 0]
        tri_y = [0, math.sin(math.radians(60)), 0, 0]
        ax.plot(tri_x, tri_y, 'k', lw=3)
        # Reference guides for II/IC/CC (geologic magenta, tectonic dashed black)
        try:
            Mseggeol = np.array([
                [29.083333, 49.472222, 85.416667, 148.416667],
                [85.416667, 148.416667, 93.527778, 136.861111],
                [93.527778, 136.861111, 85.222222, 81.611111],
                [84.861111, 81.250000, 143.361111, 14.805556],
                [143.361111, 14.805556, 137.944444, 0.000000],
                [137.944444, 0.000000, 66.083333, 0.361111],
                [66.083333, 0.361111, 29.083333, 49.472222],
            ]) / 190.0
            for row in Mseggeol:
                ax.plot([row[0], row[2]], [row[1], row[3]], 'm', lw=2)
        except Exception:
            pass
        try:
            Msegtect = np.array([
                [156.120282, 262.095969, 143.196417, 240.383877],
                [143.196417, 240.383877, 130.789507, 215.570058],
                [130.789507, 215.570058, 121.484325, 192.307102],
                [121.484325, 192.307102, 114.246961, 170.595010],
                [113.730006, 170.078055, 107.594370, 149.916827],
                [107.594370, 149.916827, 105.458733, 131.823417],
                [105.458733, 131.823417, 104.941779, 112.179143],
                [104.941779, 112.179143, 106.492642, 97.187460],
                [106.492642, 97.187460, 111.662188, 81.678823],
                [111.662188, 81.678823, 117.348688, 67.721049],
                [117.348688, 67.721049, 127.170825, 54.280230],
                [127.170825, 54.280230, 138.543826, 42.390275],
                [138.543826, 42.390275, 150.433781, 32.051184],
                [150.433781, 32.051184, 167.493282, 23.779910],
                [166.976328, 23.779910, 185.586692, 16.542546],
                [185.586692, 16.542546, 205.747921, 10.339091],
                [205.747921, 10.339091, 228.493922, 5.169546],
                [228.493922, 5.169546, 254.341651, 3.101727],
                [254.341651, 3.101727, 281.740243, 0.516955],
                [281.740243, 0.516955, 310.689699, -1.033909],
            ]) / 308.0
            for row in Msegtect:
                ax.plot([row[0], row[2]], [row[1], row[3]], '--k', lw=2)
            # Two reference dots at segments 10 and 13 end-points (x4,y4)
            ax.plot(Msegtect[9, 2], Msegtect[9, 3], '.k', ms=10)
            ax.plot(Msegtect[12, 2], Msegtect[12, 3], '.k', ms=10)
        except Exception:
            pass
        ax.plot(Txp, Typ, '.r', ms=12)
        ax.text(0.45, 0.95, 'I-I', weight='bold', ha='center')
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
            ax.text(1, 0.5*lim, f"{float(np.max(kvalnorm)):.4f}")
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
        ax.text(1, 0.4*lim, f"{float(np.max(k2valnorm)):.4f}")
        ax.text(1, 0.3*lim, f"f={f:.4f}")
        ax.set_aspect('equal', 'box')
        ax.set_title('razão de permeabilidade ii', pad=6, fontsize=10)
        ax.tick_params(labelsize=8)

        # 4,1: C_B (interpolated filled-contours similar ao MATLAB)
        ax = self.axes[3][0]
        ax.clear()
        lados = max(int(params.discretizacao), 1)
        prolatlon = (ymax - ymin) / (xmax - xmin) if (xmax > xmin) else 1.0
        nrows = max(int(round(lados * prolatlon)), 1)
        ncols = lados
        CB = np.zeros((nrows, ncols), dtype=float)
        if segs.shape[0] > 0:
            dx = (xmax - xmin) / ncols if ncols > 0 else (xmax - xmin)
            dy = (ymax - ymin) / nrows if nrows > 0 else (ymax - ymin)
            # centers for display
            lonvec = np.zeros(ncols)
            latvec = np.zeros(nrows)
            for mm in range(nrows-1, -1, -1):
                for nn in range(0, ncols):
                    xilim = xmin + nn * dx
                    xflim = xmin + (nn + 1) * dx
                    yilim = ymin + mm * dy
                    yflim = ymin + (mm + 1) * dy
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
        # Interpolação por spline de placa fina (TPS) para emular griddata 'v4' do MATLAB
        try:
            resolucao = 100
            loni = np.linspace(xmin, xmax, resolucao)
            lati = np.linspace(ymin, ymax, resolucao)
            lonx, laty = np.meshgrid(loni, lati)
            # pontos centrais das células com valores CB
            lonc, latc = np.meshgrid(lonvec, latvec)
            Xs = lonc.ravel(); Ys = latc.ravel(); Vs = CB.ravel()
            n = Xs.size
            # Montar sistema TPS
            def U_from_d2(d2):
                with np.errstate(divide='ignore', invalid='ignore'):
                    return d2 * np.where(d2 > 0, 0.5*np.log(d2), 0.0)
            d2 = (Xs[:, None]-Xs[None, :])**2 + (Ys[:, None]-Ys[None, :])**2
            K = U_from_d2(d2)
            Pmat = np.column_stack([np.ones(n), Xs, Ys])
            A = np.block([[K, Pmat], [Pmat.T, np.zeros((3,3))]])
            b = np.concatenate([Vs, np.zeros(3)])
            sol = np.linalg.lstsq(A, b, rcond=None)[0]
            w = sol[:n]
            a0, a1, a2 = sol[n:]
            # Avaliar na grade
            XI = lonx.ravel(); YI = laty.ravel()
            d2q = (XI[:, None]-Xs[None, :])**2 + (YI[:, None]-Ys[None, :])**2
            Uq = U_from_d2(d2q)
            ZI = Uq @ w + a0 + a1*XI + a2*YI
            CBint = ZI.reshape(lonx.shape)
            # limitar aos extremos de CB
            cbmin = float(np.min(CB)) if CB.size else 0.0
            cbmax = float(np.max(CB)) if CB.size else 1.0
            CBint = np.clip(CBint, cbmin, cbmax)
            # Contornos preenchidos (aproxima MATLAB contourf padrão)
            cf = ax.contourf(lonx, laty, CBint, cmap='parula')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('equal', 'box')
            try:
                cb_ax = inset_axes(
                    ax,
                    width="6%",
                    height="100%",
                    loc='lower left',
                    bbox_to_anchor=(1.06, 0.0, 1, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0,
                )
                self.fig.colorbar(cf, cax=cb_ax, orientation='vertical')
                cb_ax.tick_params(labelsize=8)
            except Exception:
                pass
        except Exception:
            # Fallback: mostrar matriz bruta caso interpolação falhe
            c = ax.imshow(CB, origin='lower', extent=[xmin, xmax, ymin, ymax], aspect='equal',
                          interpolation='nearest', cmap='parula')
            try:
                cb_ax = inset_axes(ax, width="6%", height="100%", loc='lower left',
                                   bbox_to_anchor=(1.06, 0.0, 1, 1), bbox_transform=ax.transAxes,
                                   borderpad=0)
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
            CB=CB, extent=(xmin, xmax, ymin, ymax), lonvec=lonvec, latvec=latvec,
            segs_sub=segs_sub, seg_classes_sub=seg_classes_sub,
        )

        # Manually adjust subplot spacing to avoid overlaps and fit inside GUI
        # Increase left margin to leave room for external zoom buttons
        self.fig.subplots_adjust(left=0.15, right=0.98, top=0.95, bottom=0.08,
                                 wspace=0.30, hspace=0.45)
        self.canvas.draw_idle()
        # Reposition zoom buttons and then tightly fit window to content
        self.position_zoom_buttons()
        self.set_window_to_requested_size()
        # After drawing, position zoom buttons relative to axes
        self.position_zoom_buttons()

    def _create_zoom_buttons(self):
        # Create buttons in the same order as axes are laid out
        for i in range(4):
            for j in range(2):
                rc = (i, j)
                # Parent buttons to the canvas widget so coordinates align with pixels
                btn = tk.Button(self._canvas_widget, text='Zoom', width=12,
                                command=lambda rc=rc: self.open_zoom(rc))
                # Place later in position_zoom_buttons()
                btn.place(x=0, y=0)
                self.zoom_buttons[rc] = btn

    def position_zoom_buttons(self):
        # Requires axes to be created and canvas size known
        try:
            cw = self._canvas_widget.winfo_width()
            ch = self._canvas_widget.winfo_height()
            # Extra gap to the left of each axes where the zoom buttons sit
            margin = 72  # pixels to the left of axes (avoid covering y-axis)
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

    def _ensure_parula_colormap(self):
        here = os.path.dirname(os.path.abspath(__file__))
        cmap_dir = os.path.join(here, 'colormaps')
        try:
            os.makedirs(cmap_dir, exist_ok=True)
        except Exception:
            pass
        csv_path = os.path.join(cmap_dir, 'parula.csv')
        # Generate CSV if missing: prefer native 'parula' if present, else 'viridis'
        if not os.path.isfile(csv_path):
            try:
                base = cm.get_cmap('parula', 256)
            except Exception:
                base = cm.get_cmap('viridis', 256)
            data = base(np.linspace(0, 1, 256))[:, :3]
            try:
                np.savetxt(csv_path, data, delimiter=',', header='r,g,b', comments='')
            except Exception:
                pass
        # Load CSV and register
        try:
            arr = np.loadtxt(csv_path, delimiter=',', comments='#')
            if arr.ndim == 1 and arr.size % 3 == 0:
                arr = arr.reshape((-1, 3))
            if arr.shape[1] > 3:
                arr = arr[:, :3]
            if arr.size == 0 or arr.shape[0] < 16:
                raise ValueError('parula.csv has insufficient rows')
            if arr.max() > 1.5:  # 0-255 -> 0-1
                arr = arr / 255.0
            parula_cmap = ListedColormap(arr, name='parula')
            try:
                matplotlib.colormaps.unregister('parula')
            except Exception:
                pass
            matplotlib.colormaps.register(parula_cmap, name='parula')
        except Exception:
            # fallback to viridis registration
            base = cm.get_cmap('viridis', 256)
            parula_cmap = ListedColormap(base(np.linspace(0, 1, 256)), name='parula')
            try:
                matplotlib.colormaps.unregister('parula')
            except Exception:
                pass
            matplotlib.colormaps.register(parula_cmap, name='parula')

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
        try:
            win.resizable(False, False)
        except Exception:
            pass
        fig = Figure(figsize=(9, 7), dpi=100)
        ax = fig.add_subplot(1, 1, 1)

        def finish(title):
            ax.set_title(title)
            # Embed matplotlib canvas with native navigation toolbar (zoom, pan, save, etc.)
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            # Toolbar provides zoom/pan/save without custom buttons
            try:
                toolbar = NavigationToolbar2Tk(canvas, win, pack_toolbar=False)
                toolbar.update()
                toolbar.pack(side=tk.TOP, fill=tk.X)
            except Exception:
                pass
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        r, c = rc
        if (r, c) == (0, 0):
            # nós I,Y e X
            for i in range(segs.shape[0]):
                ax.plot([xi[i], xf[i]], [yi[i], yf[i]], 'k', lw=1)
            handles = []
            if I_nodes.size:
                hI, = ax.plot(I_nodes[:, 0], I_nodes[:, 1], 'o', mfc='g', mec='g', ms=6, label='I')
                handles.append(hI)
            if Y_nodes.size:
                hY, = ax.plot(Y_nodes[:, 0], Y_nodes[:, 1], '^', mfc='r', mec='r', ms=6, label='Y')
                handles.append(hY)
            if X_nodes.size:
                hX, = ax.plot(X_nodes[:, 0], X_nodes[:, 1], 's', mfc='b', mec='b', ms=6, label='X')
                handles.append(hX)
            # Match 'todos' plot behavior: autoscale with a small margin
            ax.set_aspect('equal', 'box')
            try:
                ax.margins(x=0.05, y=0.05)
            except Exception:
                pass
            # Show legend outside on the right in the zoom plot
            if handles:
                try:
                    # Create right margin to host the external legend
                    fig.subplots_adjust(right=0.80)
                    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.06, 1.0),
                              borderaxespad=0.0, fontsize=9)
                except Exception:
                    pass
            finish('nós I,Y e X')
            return
        if (r, c) == (0, 1):
            # I,Y,X triângulo
            Tx = math.cos(math.radians(60)) - pY*math.cos(math.radians(60)) + pX/2
            Ty = math.sin(math.radians(60)) - pY*math.sin(math.radians(60)) - pX*(1/math.tan(math.radians(30)))/2
            tri_x = [0, 0.5, 1, 0]
            tri_y = [0, math.sin(math.radians(60)), 0, 0]
            ax.plot(tri_x, tri_y, 'k', lw=3)
            # Reference guides (geologic magenta, tectonic dashed black) as in MATLAB
            try:
                Mnosgeol = np.array([
                    [15.805556, 27.805556, 93.000000, 162.500000],
                    [93.000000, 162.500000, 107.250000, 126.027778],
                    [107.250000, 126.027778, 111.944444, 85.222222],
                    [111.944444, 85.222222, 92.444444, 48.750000],
                    [92.444444, 48.750000, 15.805556, 27.805556],
                ]) / 190.0
                for row in Mnosgeol:
                    ax.plot([row[0], row[2]], [row[1], row[3]], 'm', lw=2)
            except Exception:
                pass
            try:
                Mnostect = np.array([
                    [117.348688, 197.476647, 185.069738, 207.298784],
                    [95.636596, 161.806782, 204.197057, 175.764555],
                ]) / 308.0
                for row in Mnostect:
                    ax.plot([row[0], row[2]], [row[1], row[3]], '--k', lw=2)
            except Exception:
                pass
            ax.plot(Tx, Ty, '.r', ms=12)
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
            # Prefer colored subsegment view (matches MATLAB)
            segs_sub = R.get('segs_sub')
            seg_classes_sub = R.get('seg_classes_sub')
            if segs_sub is not None and segs_sub.size and seg_classes_sub is not None:
                for i in range(segs_sub.shape[0]):
                    color = dict(II='g', IC='r', CC='b')[seg_classes_sub[i]]
                    ax.plot([segs_sub[i,0], segs_sub[i,2]], [segs_sub[i,1], segs_sub[i,3]], color=color, lw=1)
            else:
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
            # Reference guides for II/IC/CC (geologic magenta, tectonic dashed black)
            try:
                Mseggeol = np.array([
                    [29.083333, 49.472222, 85.416667, 148.416667],
                    [85.416667, 148.416667, 93.527778, 136.861111],
                    [93.527778, 136.861111, 85.222222, 81.611111],
                    [84.861111, 81.250000, 143.361111, 14.805556],
                    [143.361111, 14.805556, 137.944444, 0.000000],
                    [137.944444, 0.000000, 66.083333, 0.361111],
                    [66.083333, 0.361111, 29.083333, 49.472222],
                ]) / 190.0
                for row in Mseggeol:
                    ax.plot([row[0], row[2]], [row[1], row[3]], 'm', lw=2)
            except Exception:
                pass
            try:
                Msegtect = np.array([
                    [156.120282, 262.095969, 143.196417, 240.383877],
                    [143.196417, 240.383877, 130.789507, 215.570058],
                    [130.789507, 215.570058, 121.484325, 192.307102],
                    [121.484325, 192.307102, 114.246961, 170.595010],
                    [113.730006, 170.078055, 107.594370, 149.916827],
                    [107.594370, 149.916827, 105.458733, 131.823417],
                    [105.458733, 131.823417, 104.941779, 112.179143],
                    [104.941779, 112.179143, 106.492642, 97.187460],
                    [106.492642, 97.187460, 111.662188, 81.678823],
                    [111.662188, 81.678823, 117.348688, 67.721049],
                    [117.348688, 67.721049, 127.170825, 54.280230],
                    [127.170825, 54.280230, 138.543826, 42.390275],
                    [138.543826, 42.390275, 150.433781, 32.051184],
                    [150.433781, 32.051184, 167.493282, 23.779910],
                    [166.976328, 23.779910, 185.586692, 16.542546],
                    [185.586692, 16.542546, 205.747921, 10.339091],
                    [205.747921, 10.339091, 228.493922, 5.169546],
                    [228.493922, 5.169546, 254.341651, 3.101727],
                    [254.341651, 3.101727, 281.740243, 0.516955],
                    [281.740243, 0.516955, 310.689699, -1.033909],
                ]) / 308.0
                for row in Msegtect:
                    ax.plot([row[0], row[2]], [row[1], row[3]], '--k', lw=2)
                ax.plot(Msegtect[9, 2], Msegtect[9, 3], '.k', ms=10)
                ax.plot(Msegtect[12, 2], Msegtect[12, 3], '.k', ms=10)
            except Exception:
                pass
            ax.plot(Txp, Typ, '.r', ms=12)
            ax.text(0.45, 0.95, 'I-I', weight='bold', ha='center')
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
                ax.text(1, 0.5*lim, f"{float(np.max(kvalnorm)):.4f}")
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
            ax.text(1, 0.4*lim, f"{float(np.max(k2valnorm)):.4f}")
            ax.text(1, 0.3*lim, f"f={f:.4f}")
            ax.set_aspect('equal', 'box')
            finish('razão de permeabilidade ii')
            return
        if (r, c) == (3, 0):
            # C_B com contornos preenchidos (TPS)
            try:
                lonvec = R.get('lonvec')
                latvec = R.get('latvec')
                resolucao = 100
                loni = np.linspace(xmin, xmax, resolucao)
                lati = np.linspace(ymin, ymax, resolucao)
                lonx, laty = np.meshgrid(loni, lati)
                lonc, latc = np.meshgrid(lonvec, latvec)
                Xs = lonc.ravel(); Ys = latc.ravel(); Vs = CB.ravel()
                n = Xs.size
                def U_from_d2(d2):
                    with np.errstate(divide='ignore', invalid='ignore'):
                        return d2 * np.where(d2 > 0, 0.5*np.log(d2), 0.0)
                d2 = (Xs[:, None]-Xs[None, :])**2 + (Ys[:, None]-Ys[None, :])**2
                K = U_from_d2(d2)
                Pmat = np.column_stack([np.ones(n), Xs, Ys])
                A = np.block([[K, Pmat], [Pmat.T, np.zeros((3,3))]])
                b = np.concatenate([Vs, np.zeros(3)])
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
                w = sol[:n]
                a0, a1, a2 = sol[n:]
                XI = lonx.ravel(); YI = laty.ravel()
                d2q = (XI[:, None]-Xs[None, :])**2 + (YI[:, None]-Ys[None, :])**2
                Uq = U_from_d2(d2q)
                ZI = Uq @ w + a0 + a1*XI + a2*YI
                CBint = ZI.reshape(lonx.shape)
                cbmin = float(np.min(CB)) if CB.size else 0.0
                cbmax = float(np.max(CB)) if CB.size else 1.0
                CBint = np.clip(CBint, cbmin, cbmax)
                cf = ax.contourf(lonx, laty, CBint, cmap='parula')
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_aspect('equal', 'box')
                fig.colorbar(cf, ax=ax, orientation='vertical')
            except Exception:
                cimg = ax.imshow(CB, origin='lower', extent=[xmin, xmax, ymin, ymax], aspect='equal', cmap='parula')
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

    def _sync_button_widths(self):
        """Force the three action buttons to match the 'Selecionar' width (in pixels)."""
        try:
            self.master.update_idletasks()
            if not hasattr(self, 'select_btn'):
                return
            # Measure the requested pixel width of the 'Selecionar' button
            w_select = self.select_btn.winfo_reqwidth()
            # Enforce equal min pixel width on the three columns using 'Selecionar' as target
            btns = self.btn_run.nametowidget(self.btn_run.winfo_parent())
            for col in range(3):
                try:
                    btns.grid_columnconfigure(col, minsize=w_select, weight=1, uniform='btns_col')
                except Exception:
                    pass
            # Also set a matching char width hint based on 'Selecionar' width
            try:
                f = tkfont.nametofont(self.select_btn.cget('font'))
            except Exception:
                f = tkfont.nametofont('TkDefaultFont')
            char_px = max(1, f.measure('0'))
            width_chars = max(1, int(round(w_select / char_px)))
            for b in (self.btn_run, self.btn_save, self.btn_export):
                b.configure(width=width_chars)
        except Exception:
            pass

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

    def set_window_to_requested_size(self):
        try:
            self.master.update_idletasks()
            lw = self.left.winfo_reqwidth() if hasattr(self, 'left') else 0
            cw = self._canvas_widget.winfo_reqwidth() if self._canvas_widget else 0
            lh = self.left.winfo_reqheight() if hasattr(self, 'left') else 0
            ch = self._canvas_widget.winfo_reqheight() if self._canvas_widget else 0
            total_w = lw + cw + 12
            total_h = max(lh, ch) + 12
            sw = self.master.winfo_screenwidth()
            sh = self.master.winfo_screenheight()
            total_w = max(600, min(total_w, sw - 20))
            total_h = max(400, min(total_h, sh - 60))
            self.master.geometry(f"{int(total_w)}x{int(total_h)}")
        except Exception:
            pass


def main():
    root = tk.Tk()
    app = GolintouGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
